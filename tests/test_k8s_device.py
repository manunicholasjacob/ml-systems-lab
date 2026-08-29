"""The Kubernetes device, against a kubectl that answers from a directory.

Everything here runs with no cluster, no container runtime and no network, because a
test that needs a cluster is a test that stops being run. The live cluster is exercised
separately by the contract suite in test_device_contract.py, which skips itself when
there is nothing to talk to.
"""

import json
import os
import sys

import pytest

from mlsyslab.agent import RESULT_BEGIN, RESULT_END
from mlsyslab.devices import from_config
from mlsyslab.devices.base import DeviceError
from mlsyslab.devices.k8s import K8sDevice, _sq, _tar_bytes, _untar_bytes

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_kubectl.py")


@pytest.fixture
def k8s(tmp_path, monkeypatch):
    """A K8sDevice wired to the fake kubectl, plus a handle on its pretend cluster."""
    state_dir = tmp_path / "cluster"
    state_dir.mkdir()
    monkeypatch.setenv("FAKE_KUBECTL_STATE", str(state_dir))

    def set_state(**kwargs):
        path = state_dir / "state.json"
        current = json.loads(path.read_text()) if path.exists() else {}
        current.update(kwargs)
        path.write_text(json.dumps(current))

    set_state(exists=False)
    device = from_config("fake-k8s", {
        "kind": "k8s",
        "kubectl": [sys.executable, FAKE],
        "namespace": "mlsyslab",
        "pod": "mlsyslab-fake",
        "image": "example/agent@sha256:" + "cd" * 32,
        "cpu_limit": "4",
        "memory_limit": "8Gi",
        "python": sys.executable,
        "ready_timeout_s": 30,
    })
    return device, set_state, state_dir


# ------------------------------------------------------------------------ registry

def test_registry_infers_k8s_from_an_image_or_a_pod():
    assert from_config("a", {"image": "x"}).kind == "k8s"
    assert from_config("b", {"pod": "p"}).kind == "k8s"
    assert from_config("c", {"host": "h"}).kind == "ssh"
    assert from_config("d", {}).kind == "local"


def test_explicit_kind_beats_inference():
    assert from_config("e", {"kind": "local", "image": "x"}).kind == "local"


def test_kubectl_prefix_accepts_a_bare_string():
    device = from_config("f", {"kind": "k8s", "kubectl": "kubectl"})
    assert device.kubectl_prefix == ["kubectl"]


# ------------------------------------------------------------------------ manifest

def test_manifest_carries_the_limits_that_change_the_measurement(k8s):
    device, _set_state, _dir = k8s
    manifest = device.pod_manifest()
    container = manifest["spec"]["containers"][0]

    assert manifest["kind"] == "Pod"
    assert manifest["metadata"]["namespace"] == "mlsyslab"
    assert container["resources"]["limits"] == {"cpu": "4", "memory": "8Gi"}
    assert "@sha256:" in container["image"], "the image must be pinned by digest"
    assert manifest["metadata"]["labels"]["mlsyslab/device"] == "fake-k8s"


def test_manifest_mounts_models_from_the_node_rather_than_copying_them(k8s):
    device, _set_state, _dir = k8s
    device.config["host_mounts"] = [
        {"host_path": "/srv/models", "mount_path": "/models", "read_only": True}
    ]
    manifest = device.pod_manifest()
    volume = manifest["spec"]["volumes"][0]
    mount = manifest["spec"]["containers"][0]["volumeMounts"][0]
    assert volume["hostPath"]["path"] == "/srv/models"
    assert mount["mountPath"] == "/models" and mount["readOnly"] is True


def test_manifest_needs_an_image_and_says_so(k8s):
    device, _set_state, _dir = k8s
    device.image = None
    with pytest.raises(DeviceError, match="needs an 'image'"):
        device.pod_manifest()


def test_ensure_pod_applies_the_manifest_it_generated(k8s):
    device, _set_state, state_dir = k8s
    device.ensure_pod()
    applied = json.loads((state_dir / "applied.json").read_text())
    assert applied["metadata"]["name"] == "mlsyslab-fake"
    assert applied["spec"]["containers"][0]["resources"]["limits"]["cpu"] == "4"


def test_ensure_pod_gives_up_with_the_events_when_the_pod_never_starts(k8s):
    device, set_state, _dir = k8s
    device.ready_timeout_s = 1.0
    set_state(ready=False)
    with pytest.raises(DeviceError, match="was not ready"):
        device.ensure_pod()


def test_a_failed_pod_fails_fast_rather_than_waiting_out_the_timeout(k8s):
    device, set_state, _dir = k8s
    device.ready_timeout_s = 60.0
    set_state(ready=False, phase="Failed")
    import time

    started = time.time()
    with pytest.raises(DeviceError, match="phase Failed"):
        device.ensure_pod()
    assert time.time() - started < 20


def test_create_pod_false_refuses_to_invent_one(k8s):
    device, _set_state, _dir = k8s
    device.create_pod = False
    with pytest.raises(DeviceError, match="create_pod is false"):
        device.ensure_pod()


# ---------------------------------------------------------------- transport honesty

def test_transport_check_passes_on_a_clean_transport(k8s):
    device, _set_state, _dir = k8s
    device.ensure_pod()
    device.verify_transport()          # must not raise


def test_transport_check_catches_a_transport_that_eats_quotes(k8s):
    """The WSL failure that motivated this: not an error, a different answer."""
    device, set_state, _dir = k8s
    device.ensure_pod()
    set_state(mangle_args=True)
    with pytest.raises(DeviceError) as excinfo:
        device.verify_transport()
    message = str(excinfo.value)
    assert "does not pass arguments through unchanged" in message
    assert "--exec" in message, "the error should say how to fix it"


def test_shell_quoting_survives_a_hostile_string():
    assert _sq("it's") == "'it'\\''s'"
    assert _sq("plain") == "'plain'"


# ---------------------------------------------------------------------- the agent

def test_agent_round_trip_through_the_pod(k8s):
    device, _set_state, _dir = k8s
    result = device.execute({
        "kind": "command",
        "argv": [sys.executable, "-c", "print(6*7)"],
        "sample_power": False,
    })
    assert result["status"] == "ok"
    assert "42" in result["stdout"]
    assert result.get("agent_epoch_s")


def test_a_truncated_stream_is_refused_not_parsed(k8s):
    """The failure that must never become a number."""
    device, set_state, _dir = k8s
    device.ensure_pod()
    device.sync()
    set_state(truncate_result=True)
    with pytest.raises(DeviceError) as excinfo:
        device.execute({"kind": "command", "argv": [sys.executable, "-c", "print(1)"],
                        "sample_power": False})
    message = str(excinfo.value)
    assert "truncated" in message
    assert "Refusing to parse a partial payload" in message


def test_a_dead_pod_names_itself_in_the_error(k8s):
    device, set_state, _dir = k8s
    device.ensure_pod()
    device.sync()
    set_state(exec_failure="error: unable to upgrade connection: pod does not exist")
    with pytest.raises(DeviceError) as excinfo:
        device.execute({"kind": "command", "argv": ["x"], "sample_power": False})
    assert "mlsyslab/mlsyslab-fake" in str(excinfo.value)


def test_sync_puts_the_package_where_pythonpath_will_find_it(k8s):
    device, _set_state, state_dir = k8s
    root = device.sync()
    assert root == "/opt/mlsyslab"
    landed = state_dir / "fs" / "opt" / "mlsyslab" / "mlsyslab" / "agent.py"
    assert landed.exists()
    assert not list((state_dir / "fs" / "opt" / "mlsyslab").rglob("__pycache__"))


def test_sync_is_done_once_per_session(k8s):
    device, _set_state, _dir = k8s
    assert device.sync() == device.sync()
    assert device._synced


def test_package_root_is_the_sync_target(k8s):
    device, _set_state, _dir = k8s
    assert device.package_root() == device.sync()


# ---------------------------------------------------------------------- filesystem

def test_exists_reflects_the_container_not_the_host(k8s):
    device, _set_state, state_dir = k8s
    device.ensure_pod()
    target = state_dir / "fs" / "opt" / "present.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("here")
    assert device.exists("/opt/present.txt")
    assert not device.exists("/opt/definitely-not-here.txt")


def test_resolve_expands_tilde_against_the_container_home(k8s):
    device, _set_state, _dir = k8s
    device.ensure_pod()
    assert device.resolve("~/models/x.gguf") == "/root/models/x.gguf"
    assert device.resolve("/absolute/stays") == "/absolute/stays"


def test_push_and_pull_round_trip_bytes_exactly(k8s, tmp_path):
    device, _set_state, _dir = k8s
    device.ensure_pod()
    source = tmp_path / "payload.bin"
    blob = bytes(range(256)) * 40
    source.write_bytes(blob)

    device.push(str(source), "/opt/data/payload.bin")
    back = tmp_path / "returned.bin"
    device.pull("/opt/data/payload.bin", str(back))
    assert back.read_bytes() == blob


def test_pull_refuses_an_archive_that_would_escape(tmp_path):
    import io
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        info = tarfile.TarInfo("../escaped.txt")
        info.size = 3
        tar.addfile(info, io.BytesIO(b"bad"))
    with pytest.raises(Exception):
        _untar_bytes(buffer.getvalue(), str(tmp_path / "out"))
    assert not (tmp_path / "escaped.txt").exists()


def test_tar_helper_refuses_a_missing_source(tmp_path):
    with pytest.raises(DeviceError, match="does not exist"):
        _tar_bytes(str(tmp_path / "nope"), arcname="nope", skip_pycache=False)


# --------------------------------------------------------------------- provenance

def test_device_info_records_the_pod_the_node_and_the_image_digest(k8s):
    device, _set_state, _dir = k8s
    info = device.device_info()
    assert info["placement"]["pod"] == "mlsyslab-fake"
    assert info["placement"]["node"] == "fake-node"
    assert info["placement"]["image_id"].startswith("sha256:")
    assert info["kind"] == "k8s"


def test_cgroup_limits_are_read_from_inside_the_container(k8s):
    device, set_state, _dir = k8s
    set_state(cgroup_lines=["cpu.max=400000 100000", "memory.max=8589934592",
                            "cpuset.cpus.effective=0-19"])
    device.ensure_pod()
    limits = device.cgroup_limits()
    assert limits["cgroup_version"] == 2
    assert limits["cpu.max"] == "400000 100000"
    # The number an analysis actually wants, derived rather than left as a quota pair.
    assert limits["cpu_quota_cores"] == 4.0
    assert limits["cpuset.cpus.effective"] == "0-19"


def test_an_unlimited_container_reports_no_quota(k8s):
    device, set_state, _dir = k8s
    set_state(cgroup_lines=["cpu.max=max", "memory.max=max"])
    device.ensure_pod()
    limits = device.cgroup_limits()
    assert limits["cpu.max"] == "max"
    assert limits["cpu_quota_cores"] is None


def test_deleting_the_pod_resets_what_was_cached_about_it(k8s):
    device, _set_state, _dir = k8s
    device.sync()
    assert device._synced
    device.delete_pod(wait=False)
    assert not device._synced and not device._pod_ready


# ----------------------------------------------------- no backend knows about this

def test_no_backend_mentions_kubernetes():
    """The design test from the build note, enforced rather than asserted in prose."""
    import pathlib

    backends = pathlib.Path(__file__).resolve().parents[1] / "src" / "mlsyslab" / "backends"
    for path in backends.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for forbidden in ("k8s", "kubernetes", "kubectl", "container", "pod"):
            assert forbidden not in text, (
                f"{path.name} mentions '{forbidden}'. A backend that knows which device "
                "kind it is running on has broken the abstraction this device relies on."
            )


# ------------------------------------------------ a pod must match the config it claims

def test_a_pod_whose_limits_no_longer_match_the_config_is_replaced(k8s, tmp_path):
    """Changing cpu_limit must produce a new pod, not reuse the old quota silently."""
    device, _set_state, state_dir = k8s
    device.ensure_pod()
    first = json.loads((state_dir / "applied.json").read_text())
    first_hash = first["metadata"]["annotations"]["mlsyslab/spec-hash"]

    device.config["cpu_limit"] = "2"
    device.ready_timeout_s = 30
    device._pod_ready = False              # a fresh process would not have this cached
    device.ensure_pod()

    second = json.loads((state_dir / "applied.json").read_text())
    assert second["spec"]["containers"][0]["resources"]["limits"]["cpu"] == "2"
    assert second["metadata"]["annotations"]["mlsyslab/spec-hash"] != first_hash


def test_an_unchanged_config_reuses_the_pod_that_is_already_running(k8s, tmp_path):
    device, _set_state, state_dir = k8s
    device.ensure_pod()
    applied = (state_dir / "applied.json").read_text()
    device._pod_ready = False
    device.ensure_pod()
    assert (state_dir / "applied.json").read_text() == applied


def test_a_foreign_pod_is_not_quietly_adopted(k8s, tmp_path):
    """A pod nobody here created carries no spec hash, so its limits are unknown."""
    device, set_state, state_dir = k8s
    device.ensure_pod()
    device.create_pod = False
    device._pod_ready = False
    (state_dir / "applied.json").write_text("{}")
    import mlsyslab.devices.k8s as k8smod

    original = k8smod.K8sDevice.pod_json

    def without_annotations(self):
        pod = original(self)
        if pod:
            pod.setdefault("metadata", {})["annotations"] = {}
        return pod

    k8smod.K8sDevice.pod_json = without_annotations
    try:
        with pytest.raises(DeviceError, match="carries no spec hash"):
            device.ensure_pod()
    finally:
        k8smod.K8sDevice.pod_json = original


def test_a_stale_pod_that_never_became_ready_is_still_replaced(k8s, tmp_path):
    """The Pending case. Applying at an immutable spec fails with a wall of diff."""
    device, set_state, state_dir = k8s
    device.ready_timeout_s = 1.0
    set_state(ready=False)
    with pytest.raises(DeviceError, match="was not ready"):
        device.ensure_pod()                       # leaves a pod behind, not Ready
    first = json.loads((state_dir / "applied.json").read_text())

    device.config["memory_limit"] = "4Gi"
    device._pod_ready = False
    set_state(ready=True)
    device.ready_timeout_s = 30
    device.ensure_pod()

    second = json.loads((state_dir / "applied.json").read_text())
    assert second["spec"]["containers"][0]["resources"]["limits"]["memory"] == "4Gi"
    assert (second["metadata"]["annotations"]["mlsyslab/spec-hash"]
            != first["metadata"]["annotations"]["mlsyslab/spec-hash"])


def test_placement_travels_with_the_record_not_beside_it(k8s):
    """The quota a run was taken under has to be inside the run's own record."""
    device, set_state, _dir = k8s
    set_state(cgroup_lines=["cpu.max=400000 100000", "memory.max=4294967296"])
    placement = device.device_info()["placement"]
    assert placement["runtime"] == "kubernetes"
    assert placement["pod"] == "mlsyslab-fake"
    assert placement["cpu_quota_cores"] == 4.0
    assert placement["cgroup"]["cpu.max"] == "400000 100000"
    assert placement["image_id"].startswith("sha256:")


def test_placement_reaches_the_run_record_through_the_backend(k8s):
    """End to end: a backend that knows nothing about pods still carries the quota."""
    from mlsyslab.backends.base import RunSpec
    from mlsyslab.backends.llamacpp import LlamaCppBackend
    from tests.conftest import LLAMA_BENCH_STDOUT

    device, set_state, _dir = k8s
    set_state(cgroup_lines=["cpu.max=200000 100000", "memory.max=4294967296"])
    device.ensure_pod()
    spec = RunSpec(device_id="fake-k8s", backend="llamacpp", model="m",
                   model_path="/m.gguf", threads=4)
    result = {"status": "ok", "duration_s": 1.0, "stdout": LLAMA_BENCH_STDOUT,
              "stderr": "", "agent_time_utc": "2026-08-28T18:00:00Z",
              "device": {"device_id": "fake-k8s", "kind": "k8s"},
              "telemetry": {}, "warnings": []}
    record = LlamaCppBackend().parse(spec, result, device)
    assert record.device.placement["cpu_quota_cores"] == 2.0
    assert "cpu_quota_cores" in record.to_json()


# ------------------------------------------------ the checked-in manifests must not drift

def test_checked_in_manifests_match_what_the_device_would_create(tmp_path):
    """A manifest in the repo that no longer matches the generator is worse than none.

    It describes limits that never applied to any number in the results directory, and
    the only way anyone finds out is by reading both files side by side, which nobody
    does.
    """
    import pathlib
    import subprocess

    repo = pathlib.Path(__file__).resolve().parents[1]
    before = {p.name: p.read_text(encoding="utf-8")
              for p in (repo / "k8s").glob("pod-agent-*.yaml")}
    assert before, "no generated manifests are checked in"

    done = subprocess.run([sys.executable, str(repo / "tools" / "write_manifests.py")],
                          capture_output=True, cwd=str(repo))
    assert done.returncode == 0, done.stderr.decode()

    after = {p.name: p.read_text(encoding="utf-8")
             for p in (repo / "k8s").glob("pod-agent-*.yaml")}
    drifted = [name for name in before if before[name] != after.get(name)]
    assert not drifted, (
        f"{drifted} differ from what tools/write_manifests.py produces. "
        "Regenerate them rather than editing by hand.")


def test_every_shipped_config_parses_and_names_real_devices():
    """Config files are documentation people copy. A broken one teaches the wrong thing."""
    import pathlib

    from mlsyslab.config import load
    from mlsyslab.devices import _REGISTRY

    repo = pathlib.Path(__file__).resolve().parents[1]
    checked = 0
    for path in sorted((repo / "configs").glob("*.yaml")):
        try:
            config = load(str(path))
        except ImportError:
            return                      # no PyYAML in this environment
        except Exception as exc:        # pragma: no cover - a broken config is the failure
            raise AssertionError(f"{path.name} does not load: {exc}") from exc
        assert config.specs, f"{path.name} expands to no runs"
        for device_id, block in config.devices.items():
            kind = block.get("kind") or (
                "k8s" if (block.get("pod") or block.get("image") or block.get("namespace"))
                else "ssh" if block.get("host") else "local")
            assert kind in _REGISTRY, f"{path.name}: device '{device_id}' has kind '{kind}'"
        checked += 1
    assert checked, "no configs were checked"


# --------------------------------------------- a container restart is an empty container

def test_a_restarted_container_is_re_synced_before_it_is_used(k8s, tmp_path):
    """The bug that stopped a real sweep twenty minutes in.

    The kubelet restarts the container, everything written into it since it started is
    gone, and the pod goes on reporting Running throughout. A sync cached for the life of
    the *device* then sends work to a container that no longer has the harness in it, and
    the sweep dies on ``No module named 'mlsyslab'`` across every pod at once.
    """
    device, set_state, state_dir = k8s
    device.sync()
    assert device._synced
    landed = state_dir / "fs" / "opt" / "mlsyslab" / "mlsyslab" / "agent.py"
    assert landed.exists()

    # The container restarts: new start time, higher restart count, empty filesystem.
    import shutil

    shutil.rmtree(state_dir / "fs" / "opt" / "mlsyslab")
    set_state(restart_count=1, started_at="2026-08-28T21:30:00Z")

    device.sync()
    assert landed.exists(), "a restarted container must be re-synced, not assumed"


def test_an_unrestarted_container_is_not_re_synced_every_run(k8s, tmp_path):
    """The other half: this check runs before every run and must stay cheap."""
    device, _set_state, state_dir = k8s
    device.sync()
    landed = state_dir / "fs" / "opt" / "mlsyslab" / "mlsyslab" / "agent.py"
    stamp = landed.stat().st_mtime_ns
    device.sync()
    device.sync()
    assert landed.stat().st_mtime_ns == stamp


def test_container_identity_changes_when_the_container_does(k8s):
    device, set_state, _dir = k8s
    device.ensure_pod()
    first = device.container_identity()
    set_state(restart_count=3, started_at="2026-08-28T22:00:00Z")
    assert device.container_identity() != first


def test_exists_says_it_could_not_look_rather_than_saying_no(k8s):
    """A blip during discovery must not be recorded as a missing binary.

    "llama-bench not found on k8s-cpu2" is classified as a configuration error and never
    retried, so a one second outage during backend discovery becomes a permanent hole in
    the matrix that blames the config.
    """
    device, set_state, _dir = k8s
    device.ensure_pod()
    set_state(exec_failure="error: unable to upgrade connection: container not found")
    with pytest.raises(DeviceError, match="could not determine whether"):
        device.exists("/usr/local/bin/llama-bench")


def test_a_discovery_blip_is_classified_as_transport_so_it_gets_retried(k8s, tmp_path):
    """End to end: the failure has to reach the scheduler as retryable."""
    from mlsyslab.config import Config
    from mlsyslab.runner import Runner
    from mlsyslab.backends.base import RunSpec

    device, set_state, _dir = k8s
    device.ensure_pod()
    set_state(exec_failure="error: unable to upgrade connection")

    runner = Runner(Config(experiment="t", output_dir=str(tmp_path / "out"),
                           devices={"fake-k8s": {}}), resume=False)
    runner._devices["fake-k8s"] = device
    spec = RunSpec(experiment="t", device_id="fake-k8s", backend="llamacpp",
                   model="m", model_path="/m.gguf")
    attempt = runner.attempt(spec)
    assert attempt.failure_kind == "transport"
    assert attempt.retryable


def test_binary_discovery_asks_once_rather_than_once_per_candidate(k8s, monkeypatch):
    """Sixteen SSH handshakes per run is how the sshd rate limiter got tripped."""
    device, _set_state, state_dir = k8s
    device.ensure_pod()
    target = state_dir / "fs" / "usr" / "local" / "bin"
    target.mkdir(parents=True, exist_ok=True)
    (target / "llama-bench").write_text("#!/bin/sh\n")

    calls = []
    original = device._sh

    def counted(script, timeout_s=120):
        calls.append(script)
        return original(script, timeout_s)

    monkeypatch.setattr(device, "_sh", counted)
    found = device.find_binary(["llama-bench", "llama-bench.exe"],
                               ["/nope/one", "/nope/two", "/usr/local/bin", "/nope/three"])
    assert found == "/usr/local/bin/llama-bench"
    assert len(calls) == 1, f"searched with {len(calls)} round trips, expected 1"


def test_a_search_that_finds_nothing_is_not_a_search_that_failed(k8s):
    device, _set_state, _dir = k8s
    device.ensure_pod()
    assert device.find_binary(["nothing-here"], ["/nope"]) is None


def test_a_search_that_could_not_run_raises_rather_than_returning_none(k8s):
    device, set_state, _dir = k8s
    device.ensure_pod()
    set_state(exec_failure="error: unable to upgrade connection")
    with pytest.raises(DeviceError, match="could not search"):
        device.find_binary(["llama-bench"], ["/usr/local/bin"])


def test_a_pod_that_will_not_answer_is_not_blamed_on_quoting(k8s):
    """Two failures, two messages.

    An exec that failed outright is not evidence about argument quoting, and reporting it
    as such sends the reader off to fix a transport that was working.
    """
    device, set_state, _dir = k8s
    device.ensure_pod()
    set_state(exec_failure="error: unable to upgrade connection: pod does not exist")
    with pytest.raises(DeviceError) as excinfo:
        device.verify_transport()
    message = str(excinfo.value)
    assert "could not reach" in message
    assert "--exec" not in message, "a dead pod must not be diagnosed as a quoting bug"
