"""One suite, every device kind. This is what "K8sDevice is a real device" means.

The claim the Kubernetes device makes is not that it works, it is that it is
interchangeable with the local and SSH devices from a backend's point of view. That claim
is only worth anything if the same tests are pointed at all of them, so the tests live
here once and the devices are parameters.

Three targets are always available: the local machine, and two Kubernetes devices backed
by the fake kubectl in ``fake_kubectl.py``. Real remote targets join in when the
environment says how to reach them, and are skipped otherwise rather than failing on a
laptop with no cluster:

    MLSYSLAB_TEST_K8S   JSON device config for a live Kubernetes device
    MLSYSLAB_TEST_SSH   JSON device config for a live SSH device
"""

import json
import os
import sys

import pytest

from mlsyslab.devices import from_config
from mlsyslab.devices.base import DeviceError

FAKE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_kubectl.py")


def _fake_k8s_config(state_dir, **overrides):
    config = {
        "kind": "k8s",
        "kubectl": [sys.executable, FAKE],
        "namespace": "mlsyslab",
        "pod": "mlsyslab-fake",
        "image": "example/agent@sha256:" + "cd" * 32,
        "python": sys.executable,
        "ready_timeout_s": 30,
    }
    config.update(overrides)
    return config


def _targets():
    """(id, builder) for every device kind this machine can actually reach."""
    targets = [("local", lambda tmp: from_config("local", {"python": sys.executable}))]

    def build_fake(tmp, **overrides):
        state = tmp / "cluster"
        state.mkdir(exist_ok=True)
        (state / "state.json").write_text(json.dumps({"exists": False}))
        os.environ["FAKE_KUBECTL_STATE"] = str(state)
        return from_config("fake-k8s", _fake_k8s_config(state, **overrides))

    targets.append(("k8s-fake", build_fake))
    # The same device with a CPU quota, because a limited container is the case the
    # containerization study actually runs and it must satisfy the contract too.
    targets.append(("k8s-fake-limited",
                    lambda tmp: build_fake(tmp, cpu_limit="2", memory_limit="2Gi")))

    for name, variable in (("k8s-live", "MLSYSLAB_TEST_K8S"),
                           ("ssh-live", "MLSYSLAB_TEST_SSH")):
        raw = os.environ.get(variable)
        if raw:
            config = json.loads(raw)
            targets.append((name, lambda tmp, c=config, n=name: from_config(n, c)))
    return targets


TARGETS = _targets()


@pytest.fixture(params=[t[0] for t in TARGETS])
def device(request, tmp_path):
    builder = dict(TARGETS)[request.param]
    built = builder(tmp_path)
    yield built


def _scratch(device, name, tmp_path):
    """A path this device can write to, named using only the Device interface.

    Not a hardcoded directory: the SSH device works out of ``~/.mlsyslab`` under a
    non-root account and the Kubernetes device out of ``/opt/mlsyslab``. A test that
    picks one of those has stopped testing the interface and started testing a guess.
    """
    if device.kind == "local":
        return str(tmp_path / name)
    return device.package_root().rstrip("/") + "/" + name


def _place_model(device, tmp_path):
    """Put a stand-in model file where this device can see it, and return its path."""
    source = tmp_path / "m.onnx"
    source.write_bytes(b"not really an onnx graph")
    if device.kind == "local":
        return str(source)
    remote = _scratch(device, "contract-model.onnx", tmp_path)
    device.push(str(source), remote)
    return remote


# ------------------------------------------------------------------- the contract

def test_execute_runs_a_command_and_returns_its_output(device):
    result = device.execute({
        "kind": "command",
        "argv": [device.python_executable, "-c", "print(6*7)"],
        "sample_power": False,
    })
    assert result["status"] == "ok"
    assert "42" in result["stdout"]
    assert result["duration_s"] > 0


def test_a_nonzero_exit_is_a_failed_result_not_an_exception(device):
    result = device.execute({
        "kind": "command",
        "argv": [device.python_executable, "-c",
                 "import sys; sys.stderr.write('bad'); sys.exit(3)"],
        "sample_power": False,
    })
    assert result["status"] == "failed"
    assert "exit code 3" in result["error"]
    assert "bad" in result["stderr"]


def test_a_missing_binary_is_a_failed_result_not_a_crash(device):
    result = device.execute({"kind": "command",
                             "argv": ["definitely-not-a-binary-xyz"],
                             "sample_power": False})
    assert result["status"] == "failed"


def test_every_result_carries_a_clock_the_host_can_compare(device):
    result = device.execute({"kind": "sysinfo"})
    assert result["agent_time_utc"].endswith("Z")
    assert isinstance(result["agent_epoch_s"], float)


def test_probe_describes_the_machine_and_what_it_can_measure(device):
    info = device.device_info()
    assert info["device_id"] == device.device_id
    assert info["kind"] == device.kind
    assert isinstance(info["capabilities"], dict)
    assert isinstance(device.capabilities(), dict)


def test_probe_is_cached_until_asked_to_refresh(device):
    first = device.probe()
    assert device.probe() is first


def test_package_root_is_somewhere_mlsyslab_imports_from(device):
    root = device.package_root()
    assert root
    result = device.execute({
        "kind": "command",
        "argv": [device.python_executable, "-c",
                 "import mlsyslab; print('import-ok')"],
        "env": {"PYTHONPATH": root},
        "sample_power": False,
    })
    assert result["status"] == "ok", result.get("stderr", "")[-500:]
    assert "import-ok" in result["stdout"]


def test_resolve_leaves_an_absolute_path_alone(device):
    absolute = "C:/x/y" if device.kind == "local" and os.name == "nt" else "/x/y"
    assert device.resolve(absolute) == absolute


def test_exists_is_true_for_something_present_and_false_for_nonsense(device, tmp_path):
    assert not device.exists("/definitely/not/here/at/all-xyz")
    source = tmp_path / "contract-probe.txt"
    source.write_text("x")
    target = _scratch(device, "contract-probe.txt", tmp_path)
    device.push(str(source), target)
    assert device.exists(target)


def test_push_then_pull_returns_the_same_bytes(device, tmp_path):
    blob = bytes(range(256)) * 16
    source = tmp_path / "out.bin"
    source.write_bytes(blob)
    remote = _scratch(device, "contract.bin", tmp_path)
    back = tmp_path / "back.bin"

    device.push(str(source), remote)
    device.pull(remote, str(back))
    assert back.read_bytes() == blob


def test_find_binary_returns_none_rather_than_guessing(device):
    assert device.find_binary(["no-such-binary-xyz"], ["/nowhere"]) is None


def test_the_result_fence_survives_noise_on_the_stream(device):
    """Every transport mixes in banners. The payload must still come back whole."""
    script = ("import sys;"
              "sys.stderr.write('warning: some library is unhappy\\n');"
              "print('payload-marker')")
    result = device.execute({"kind": "command",
                             "argv": [device.python_executable, "-c", script],
                             "sample_power": False})
    assert result["status"] == "ok"
    assert "payload-marker" in result["stdout"]
    assert "unhappy" in result["stderr"]


def test_repr_names_the_class_and_the_device(device):
    assert device.device_id in repr(device)


# ------------------------------------------------------- what the contract is for

def test_a_backend_can_drive_any_device_without_knowing_which(device, tmp_path):
    """The whole point, stated as a test.

    The ONNX Runtime backend builds its task purely from the Device interface. If it can
    build a task for a Kubernetes pod without a single branch, the abstraction holds.
    """
    from mlsyslab.backends.base import RunSpec
    from mlsyslab.backends.onnxrt import OnnxRuntimeBackend

    model_path = _place_model(device, tmp_path)
    spec = RunSpec(device_id=device.device_id, backend="onnxruntime", model="m",
                   model_path=model_path, batch_size=1, repetitions=2)
    task = OnnxRuntimeBackend().build_task(spec, device)

    assert task["kind"] == "command"
    assert task["argv"][0] == device.python_executable
    assert task["env"]["PYTHONPATH"] == device.package_root()
