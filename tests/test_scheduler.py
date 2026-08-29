"""The concurrent scheduler, including the failure modes it exists to survive.

Every device here is scripted rather than real. The point of these tests is the
scheduler's behaviour when a machine dies, and a test that needed a machine to actually
die would be a test nobody ever runs.
"""

import json
import os
import threading
import time

import pytest

from mlsyslab import backends as backend_registry
from mlsyslab.backends.base import Backend, RunSpec
from mlsyslab.config import Config
from mlsyslab.devices.base import DeviceError
from mlsyslab.runner import Runner
from mlsyslab.scheduler import (ConcurrentSweep, DevicePolicy, RetryPolicy, _Breaker,
                                _respec_for_device)
from mlsyslab.schema import BackendInfo, compute_run_id

from tests.conftest import FakeDevice, LLAMA_BENCH_STDOUT

GOOD_RESULT = {
    "status": "ok", "duration_s": 1.0, "stdout": LLAMA_BENCH_STDOUT, "stderr": "",
    "agent_time_utc": "2026-08-28T18:00:00Z", "agent_epoch_s": None,
    "device": {"device_id": "x", "kind": "local"}, "telemetry": {}, "warnings": [],
}


class PassthroughBackend(Backend):
    name = "sched-test"

    def discover(self, device):
        return BackendInfo(name=self.name)

    def build_task(self, spec, device):
        return {"kind": "command", "argv": ["fake"]}

    def parse(self, spec, result, device):
        from mlsyslab.backends.llamacpp import LlamaCppBackend

        return LlamaCppBackend().parse(spec, result, device)


class ScriptedDevice(FakeDevice):
    """A device that can be made slow, made to die, and asked what it did.

    ``fail_after`` makes it start raising DeviceError once that many calls have
    succeeded, which is how a machine that dies halfway through a sweep is simulated.
    """

    # Across every ScriptedDevice, not just this one: the question a resource group
    # answers is "was anything else on the same box running at the same time".
    global_lock = threading.Lock()
    global_now = 0
    global_peak = 0

    @classmethod
    def reset_witness(cls):
        with cls.global_lock:
            cls.global_now = 0
            cls.global_peak = 0

    def __init__(self, device_id, delay=0.0, fail_after=None, fail_message="link went away"):
        super().__init__(device_id=device_id, result=dict(GOOD_RESULT))
        self.delay = delay
        self.fail_after = fail_after
        self.fail_message = fail_message
        self.calls = 0
        self.successes = 0
        self.alive = True
        self.concurrent_now = 0
        self.concurrent_peak = 0
        self._lock = threading.Lock()

    def execute(self, task, timeout_s=None):
        with self._lock:
            self.calls += 1
            self.concurrent_now += 1
            self.concurrent_peak = max(self.concurrent_peak, self.concurrent_now)
        measured = task.get("kind") != "sysinfo"
        if measured:
            with ScriptedDevice.global_lock:
                ScriptedDevice.global_now += 1
                ScriptedDevice.global_peak = max(ScriptedDevice.global_peak,
                                                 ScriptedDevice.global_now)
        try:
            if self.delay:
                time.sleep(self.delay)
            if task.get("kind") == "sysinfo":
                return {"status": "ok", "agent_epoch_s": time.time(),
                        "device": {"device_id": self.device_id}, "capabilities": {}}
            if not self.alive or (self.fail_after is not None
                                  and self.successes >= self.fail_after):
                raise DeviceError(self.fail_message)
            with self._lock:
                self.successes += 1
            return dict(self.result)
        finally:
            with self._lock:
                self.concurrent_now -= 1
            if measured:
                with ScriptedDevice.global_lock:
                    ScriptedDevice.global_now -= 1


def make_config(tmp_path, devices):
    backend_registry.register("sched-test", PassthroughBackend)
    return Config(experiment="sched", output_dir=str(tmp_path / "out"), devices=devices)


def specs_for(device_id, count, start=0):
    return [RunSpec(experiment="sched", device_id=device_id, backend="sched-test",
                    model="m", model_path="/m.gguf", threads=1 + start + i,
                    prompt_tokens=128, output_tokens=64)
            for i in range(count)]


def fast_retry(**kwargs):
    defaults = dict(max_attempts=2, backoff_s=0.01, backoff_factor=1.0,
                    max_backoff_s=0.05, jitter=0.0)
    defaults.update(kwargs)
    return RetryPolicy(**defaults)


def wire(tmp_path, devices_config, device_objects, **sweep_kwargs):
    ScriptedDevice.reset_witness()
    config = make_config(tmp_path, devices_config)
    runner = Runner(config, resume=True)
    for device in device_objects:
        runner._devices[device.device_id] = device
    sweep_kwargs.setdefault("retry", fast_retry())
    sweep_kwargs.setdefault("rng_seed", 0)
    return runner, ConcurrentSweep(runner, **sweep_kwargs)


# --------------------------------------------------------------------- happy path

def test_concurrent_sweep_covers_three_devices(tmp_path):
    devices = [ScriptedDevice("laptop"), ScriptedDevice("pi5"), ScriptedDevice("k8s")]
    runner, sweep = wire(
        tmp_path,
        {"laptop": {"max_concurrency": 3}, "pi5": {"max_concurrency": 1},
         "k8s": {"max_concurrency": 2}},
        devices,
    )
    specs = specs_for("laptop", 4) + specs_for("pi5", 2, 100) + specs_for("k8s", 3, 200)

    result = sweep.run(specs, check_clocks=False)

    assert len(result.records) == 9
    assert all(r.ok for r in result.records)
    assert result.summary["by_device"]["laptop"]["ok"] == 4
    assert result.summary["by_device"]["pi5"]["ok"] == 2
    assert result.summary["by_device"]["k8s"]["ok"] == 3
    for spec in specs:
        assert os.path.exists(runner.record_path(spec))


def test_per_device_concurrency_is_respected(tmp_path):
    """The Pi gets exactly one job at a time however busy the sweep is."""
    laptop = ScriptedDevice("laptop", delay=0.05)
    pi = ScriptedDevice("pi5", delay=0.05)
    runner, sweep = wire(
        tmp_path,
        {"laptop": {"max_concurrency": 4}, "pi5": {"max_concurrency": 1}},
        [laptop, pi],
    )
    sweep.run(specs_for("laptop", 8) + specs_for("pi5", 4, 100), check_clocks=False)

    assert pi.concurrent_peak == 1, "the 2 GB board must never be given two jobs at once"
    assert laptop.concurrent_peak > 1, "the laptop should actually have run in parallel"
    assert laptop.concurrent_peak <= 4


def test_concurrency_defaults_to_one_when_unstated(tmp_path):
    device = ScriptedDevice("unknown-box", delay=0.02)
    runner, sweep = wire(tmp_path, {"unknown-box": {}}, [device])
    sweep.run(specs_for("unknown-box", 4), check_clocks=False)
    assert device.concurrent_peak == 1


# ------------------------------------------------------------- retries and their record

def test_transport_failure_is_retried_and_written_down(tmp_path):
    class FlakyOnce(ScriptedDevice):
        def execute(self, task, timeout_s=None):
            if task.get("kind") != "sysinfo" and self.calls == 0:
                self.calls += 1
                raise DeviceError("tailnet blip")
            return super().execute(task, timeout_s)

    device = FlakyOnce("pi5")
    runner, sweep = wire(tmp_path, {"pi5": {}}, [device])
    result = sweep.run(specs_for("pi5", 1), check_clocks=False)

    assert len(result.records) == 1
    assert result.records[0].ok
    # The number is usable, but it does not get to pretend it came first time.
    assert any("attempt 2" in w for w in result.records[0].warnings)
    assert any(e["event"] == "transport_failure" for e in result.retries)

    ledger = os.path.join(runner.output_dir, "retries.jsonl")
    assert os.path.exists(ledger)
    with open(ledger, encoding="utf-8") as fh:
        entries = [json.loads(line) for line in fh if line.strip()]
    assert entries[0]["device"] == "pi5"
    assert "tailnet blip" in entries[0]["error"]


def test_measurement_failure_is_not_retried(tmp_path):
    """A model that will not fit is a finding. Retrying it would bury the finding."""
    device = ScriptedDevice("laptop")
    device.result = {"status": "failed", "error": "not enough memory", "stdout": "",
                     "stderr": "", "device": {}, "telemetry": {}, "warnings": []}
    runner, sweep = wire(tmp_path, {"laptop": {}}, [device])
    result = sweep.run(specs_for("laptop", 1), check_clocks=False)

    assert not result.records[0].ok
    assert device.calls == 1, "a failed measurement must be measured exactly once"
    assert not [e for e in result.retries if e["event"] == "transport_failure"]


def test_backoff_grows_and_is_bounded():
    policy = RetryPolicy(backoff_s=1.0, backoff_factor=2.0, max_backoff_s=5.0, jitter=0.0)
    assert policy.delay_for(2) == 1.0
    assert policy.delay_for(3) == 2.0
    assert policy.delay_for(4) == 4.0
    assert policy.delay_for(9) == 5.0        # bounded, not unbounded exponential


# ------------------------------------------------------------------- circuit breaker

def test_breaker_opens_then_half_opens_then_closes():
    policy = DevicePolicy("pi5", max_transport_failures=2, recovery_after_s=10.0)
    breaker = _Breaker(policy)

    assert breaker.allow(0.0)
    assert breaker.record_failure("boom", 0.0) is False     # one failure is not a verdict
    assert breaker.record_failure("boom", 1.0) is True      # two consecutive is
    assert breaker.is_open
    assert not breaker.allow(2.0)                            # inside the cooldown

    assert breaker.allow(11.5)                               # half open: one probe allowed
    assert not breaker.allow(11.6)                           # and only one
    breaker.record_failure("still boom", 12.0)
    assert breaker.is_open
    assert breaker.down_since == 1.0, "down_since must survive a failed probe"

    assert breaker.allow(23.0)
    assert breaker.record_success() is True
    assert not breaker.is_open
    assert breaker.down_since is None
    assert breaker.recoveries == 1


# ------------------------------------------------- a device dies in the middle of a sweep

def test_device_killed_mid_sweep_does_not_lose_the_run(tmp_path):
    """The requirement from the design note, tested end to end.

    A machine dies after two points. The sweep must finish, the healthy machine's work
    must be complete and correct, and every unmeasured point on the dead machine must
    exist on disk as a failed record that says why, so the hole is visible and a resumed
    sweep can pick it up.
    """
    laptop = ScriptedDevice("laptop", delay=0.01)
    pi = ScriptedDevice("pi5", delay=0.01, fail_after=2, fail_message="ssh: connection reset")

    runner, sweep = wire(
        tmp_path,
        {"laptop": {"max_concurrency": 2},
         "pi5": {"max_concurrency": 1, "max_transport_failures": 1,
                 "recovery_after_s": 0.05}},
        [laptop, pi],
        retry=fast_retry(max_attempts=2),
    )
    laptop_specs = specs_for("laptop", 5)
    pi_specs = specs_for("pi5", 8, 100)

    started = time.time()
    result = sweep.run(laptop_specs + pi_specs, check_clocks=False)
    elapsed = time.time() - started

    assert elapsed < 60, "a dead device must not stall the sweep"

    # The healthy machine finished all of its work.
    for spec in laptop_specs:
        with open(runner.record_path(spec), encoding="utf-8") as fh:
            assert json.load(fh)["status"] == "ok"

    # Every point on the dead machine exists on disk, and none of them claims success.
    states = []
    for spec in pi_specs:
        path = runner.record_path(spec)
        assert os.path.exists(path), f"no record written for {spec.threads}"
        with open(path, encoding="utf-8") as fh:
            states.append(json.load(fh))
    assert [s["status"] for s in states].count("ok") == 2      # the two before it died
    failed = [s for s in states if s["status"] == "failed"]
    assert len(failed) == 6
    assert all(s.get("error") for s in failed), "a failure with no reason is not a record"

    # At least one point was never attempted at all, and says so rather than implying
    # that it was measured and came back empty.
    assert any("not attempted" in s["error"] for s in failed)

    assert sweep._breakers["pi5"].trips >= 1
    assert result.summary["breakers"]["pi5"]["trips"] >= 1
    assert os.path.exists(os.path.join(runner.output_dir, "retries.jsonl"))
    assert os.path.exists(os.path.join(runner.output_dir, "schedule.json"))


def test_resume_after_a_killed_device_repeats_no_work(tmp_path):
    """The second half of the same requirement: restart, and do only what is missing."""
    laptop = ScriptedDevice("laptop", delay=0.01)
    pi = ScriptedDevice("pi5", delay=0.01, fail_after=1, fail_message="pod deleted")
    devices_config = {"laptop": {"max_concurrency": 2},
                      "pi5": {"max_concurrency": 1, "max_transport_failures": 1,
                              "recovery_after_s": 0.05}}
    runner, sweep = wire(tmp_path, devices_config, [laptop, pi],
                         retry=fast_retry(max_attempts=2))
    laptop_specs = specs_for("laptop", 4)
    pi_specs = specs_for("pi5", 5, 100)
    sweep.run(laptop_specs + pi_specs, check_clocks=False)

    laptop_calls_after_first = laptop.successes
    pi_ok_after_first = sum(
        1 for spec in pi_specs
        if json.load(open(runner.record_path(spec), encoding="utf-8"))["status"] == "ok"
    )
    assert pi_ok_after_first < len(pi_specs)

    # The Pi comes back. Same output directory, same specs.
    healed_laptop = ScriptedDevice("laptop", delay=0.01)
    healed_pi = ScriptedDevice("pi5", delay=0.01)
    runner2 = Runner(make_config(tmp_path, devices_config), resume=True)
    runner2._devices["laptop"] = healed_laptop
    runner2._devices["pi5"] = healed_pi
    sweep2 = ConcurrentSweep(runner2, retry=fast_retry(), rng_seed=0)
    result2 = sweep2.run(laptop_specs + pi_specs, check_clocks=False)

    assert healed_laptop.calls == 0, "completed laptop points must not be measured again"
    assert healed_pi.successes == len(pi_specs) - pi_ok_after_first
    assert result2.summary["skipped_already_done"] == len(laptop_specs) + pi_ok_after_first
    assert all(r.ok for r in result2.records)

    # And now the whole matrix is complete, with exactly one record per point.
    for spec in laptop_specs + pi_specs:
        with open(runner.record_path(spec), encoding="utf-8") as fh:
            assert json.load(fh)["status"] == "ok"
    assert laptop_calls_after_first == len(laptop_specs)


# ------------------------------------------------------------------------- failover

def test_no_failover_unless_the_config_declares_it(tmp_path):
    """The default must be to leave a hole, not to quietly measure something else."""
    laptop = ScriptedDevice("laptop")
    pi = ScriptedDevice("pi5", fail_after=0, fail_message="board is off")
    runner, sweep = wire(
        tmp_path,
        {"laptop": {}, "pi5": {"max_transport_failures": 1, "recovery_after_s": 0.05}},
        [laptop, pi],
        retry=fast_retry(max_attempts=1),
    )
    sweep.run(specs_for("pi5", 3), check_clocks=False)

    assert laptop.calls == 0, "Pi work must never silently become laptop work"
    assert not [e for e in sweep._retry_log if e["event"] == "failover"]


def test_declared_failover_moves_work_and_labels_it(tmp_path):
    laptop = ScriptedDevice("laptop")
    pi = ScriptedDevice("pi5", fail_after=0, fail_message="board is off")
    config = make_config(tmp_path, {
        "laptop": {},
        "pi5": {"max_transport_failures": 1, "recovery_after_s": 0.05,
                "failover_to": ["laptop"]},
    })
    config.models = {"m": {"paths": {"pi5": "/pi/m.gguf", "laptop": "/laptop/m.gguf"}}}
    runner = Runner(config, resume=True)
    runner._devices["laptop"] = laptop
    runner._devices["pi5"] = pi
    sweep = ConcurrentSweep(runner, retry=fast_retry(max_attempts=1), rng_seed=0)

    result = sweep.run(specs_for("pi5", 2), check_clocks=False)

    moved = [e for e in result.retries if e["event"] == "failover"]
    assert moved, "a config that declares interchangeable devices should move the work"
    assert moved[0]["from_device"] == "pi5" and moved[0]["to_device"] == "laptop"
    # The moved run is a different measurement and carries a different id.
    assert moved[0]["run_id"] != moved[0]["moved_run_id"]

    relocated = [r for r in result.records
                 if r.device.device_id == "laptop" and r.ok]
    assert relocated, "the moved point should actually have run on the laptop"
    assert any(t.startswith("failover-from:") for t in relocated[0].tags)

    # The original device's point is still recorded as failed: the hole stays visible.
    assert [r for r in result.records if r.device.device_id == "pi5" and not r.ok]


def test_a_point_with_no_path_on_the_target_cannot_be_moved():
    config = Config(models={"m": {"paths": {"pi5": "/pi/m.gguf"}}})
    spec = RunSpec(device_id="pi5", model="m", model_path="/pi/m.gguf")
    assert _respec_for_device(spec, "laptop", config) is None

    config.models["m"]["paths"]["laptop"] = "/laptop/m.gguf"
    moved = _respec_for_device(spec, "laptop", config)
    assert moved is not None
    assert moved.model_path == "/laptop/m.gguf"
    assert compute_run_id(moved.identity()) != compute_run_id(spec.identity())


# ---------------------------------------------------------------------- clock skew

def test_a_slow_agent_is_not_mistaken_for_a_wrong_clock(tmp_path):
    """The bug that motivated Cristian's algorithm here.

    Four devices on one physical machine share one clock, so the true offset is zero.
    An estimator that used only the agent's finishing stamp reported them two to three
    seconds apart, because collecting sysinfo takes that long. The agent's own runtime
    must cancel out.
    """
    device = ScriptedDevice("slow-but-punctual")

    def slow_but_punctual(task, timeout_s=None):
        started = time.time()
        time.sleep(0.4)                      # the agent doing its own work
        return {"status": "ok", "agent_started_epoch_s": started,
                "agent_epoch_s": time.time(), "device": {}, "capabilities": {}}

    device.execute = slow_but_punctual
    runner, sweep = wire(tmp_path, {"slow-but-punctual": {}}, [device])
    offsets = sweep.check_clocks(["slow-but-punctual"])

    entry = offsets["slow-but-punctual"]
    assert entry["agent_runtime_s"] >= 0.3
    assert abs(entry["offset_s"]) < 0.1, (
        f"a slow agent was read as a {entry['offset_s']:+.3f} s clock error")
    assert "warning" not in entry
    # The uncertainty must not swallow the agent's runtime either: what is unknown is
    # the transport, and that is all it may claim.
    assert entry["uncertainty_s"] < 0.2


def test_an_agent_without_a_start_stamp_still_produces_an_answer(tmp_path):
    """Older agents, and any device whose package has not been re-synced yet."""
    device = ScriptedDevice("old-agent")
    device.execute = lambda task, timeout_s=None: {
        "status": "ok", "agent_epoch_s": time.time(), "device": {}, "capabilities": {}}
    runner, sweep = wire(tmp_path, {"old-agent": {}}, [device])
    entry = sweep.check_clocks(["old-agent"])["old-agent"]
    assert entry["status"] == "ok"
    assert entry["agent_runtime_s"] == 0.0


def test_clock_check_records_the_offset_and_flags_a_bad_one(tmp_path):
    good = ScriptedDevice("laptop")
    skewed = ScriptedDevice("pi5")

    def skewed_execute(task, timeout_s=None):
        now = time.time() + 400.0
        return {"status": "ok", "agent_started_epoch_s": now, "agent_epoch_s": now,
                "device": {}, "capabilities": {}}

    skewed.execute = skewed_execute

    runner, sweep = wire(tmp_path, {"laptop": {}, "pi5": {}}, [good, skewed])
    offsets = sweep.check_clocks(["laptop", "pi5"])

    assert offsets["laptop"]["status"] == "ok"
    assert abs(offsets["laptop"]["offset_s"]) < 2.0
    assert "warning" not in offsets["laptop"]
    assert offsets["pi5"]["offset_s"] > 300
    assert "NTP" in offsets["pi5"]["warning"]


def test_unreachable_device_at_clock_check_is_reported_not_fatal(tmp_path):
    dead = ScriptedDevice("pi5", fail_after=0, fail_message="no route to host")
    runner, sweep = wire(tmp_path, {"pi5": {}}, [dead])
    offsets = sweep.check_clocks(["pi5"])
    assert offsets["pi5"]["status"] in ("ok", "unreachable")


# ------------------------------------------------------------------------- sidecars

def test_schedule_summary_is_written_and_self_describing(tmp_path):
    device = ScriptedDevice("laptop")
    runner, sweep = wire(tmp_path, {"laptop": {"max_concurrency": 2}}, [device])
    sweep.run(specs_for("laptop", 3), check_clocks=False)

    with open(os.path.join(runner.output_dir, "schedule.json"), encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["mode"] == "concurrent"
    assert summary["executed"] == 3
    assert summary["concurrency"]["laptop"] == 2
    assert summary["retry_policy"]["retried_only"] == "transport failures"


def test_sequential_and_concurrent_produce_the_same_records(tmp_path):
    """Concurrency is a scheduling change, not a measurement change."""
    sequential_device = ScriptedDevice("laptop")
    config = make_config(tmp_path / "seq", {"laptop": {}})
    sequential = Runner(config, resume=True)
    sequential._devices["laptop"] = sequential_device
    sequential_records = sequential.run(specs_for("laptop", 3))

    concurrent_device = ScriptedDevice("laptop")
    runner, sweep = wire(tmp_path / "conc", {"laptop": {"max_concurrency": 3}},
                         [concurrent_device])
    concurrent_records = sweep.run(specs_for("laptop", 3), check_clocks=False).records

    def fingerprint(records):
        return sorted((r.run_id, r.status, r.metrics.decode_tps) for r in records)

    assert fingerprint(sequential_records) == fingerprint(concurrent_records)


@pytest.mark.parametrize("limit", [1, 2, 5])
def test_sweep_terminates_for_any_limit(tmp_path, limit):
    device = ScriptedDevice("laptop", delay=0.01)
    runner, sweep = wire(tmp_path / str(limit), {"laptop": {"max_concurrency": limit}},
                         [device])
    result = sweep.run(specs_for("laptop", 6), check_clocks=False)
    assert len(result.records) == 6


# ------------------------------------------------- devices that are the same machine

def test_a_resource_group_serialises_devices_that_share_hardware(tmp_path):
    """A host and the pods scheduled onto it are three device ids and one CPU.

    Running them at once would not measure orchestration overhead, it would measure three
    processes fighting over the same cores, which is a different and much less useful
    experiment. This is the test that makes the containerization study valid.
    """
    host = ScriptedDevice("wsl-host", delay=0.05)
    pod_a = ScriptedDevice("k8s-unlimited", delay=0.05)
    pod_b = ScriptedDevice("k8s-cpu4", delay=0.05)
    shared = {"resource_group": "laptop-cpu"}
    runner, sweep = wire(
        tmp_path,
        {"wsl-host": dict(shared), "k8s-unlimited": dict(shared), "k8s-cpu4": dict(shared)},
        [host, pod_a, pod_b],
    )

    result = sweep.run(
        specs_for("wsl-host", 3) + specs_for("k8s-unlimited", 3, 100)
        + specs_for("k8s-cpu4", 3, 200),
        check_clocks=False,
    )

    assert len(result.records) == 9
    assert all(r.ok for r in result.records)
    assert ScriptedDevice.global_peak == 1, (
        "devices in one resource group must never run at the same time")
    assert result.summary["resource_groups"]["laptop-cpu"]["limit"] == 1
    assert result.summary["resource_groups"]["laptop-cpu"]["devices"] == [
        "k8s-cpu4", "k8s-unlimited", "wsl-host"]


def test_devices_in_different_groups_still_overlap(tmp_path):
    """The converse. Separate machines are the whole reason for running concurrently."""
    laptop = ScriptedDevice("laptop", delay=0.05)
    pi = ScriptedDevice("pi5", delay=0.05)
    runner, sweep = wire(tmp_path, {"laptop": {}, "pi5": {}}, [laptop, pi])
    sweep.run(specs_for("laptop", 4) + specs_for("pi5", 4, 100), check_clocks=False)
    assert ScriptedDevice.global_peak > 1


def test_a_group_can_be_told_it_holds_more_than_one(tmp_path):
    a = ScriptedDevice("pod-a", delay=0.05)
    b = ScriptedDevice("pod-b", delay=0.05)
    runner, sweep = wire(
        tmp_path,
        {"pod-a": {"resource_group": "big-box", "group_max_concurrency": 2},
         "pod-b": {"resource_group": "big-box", "group_max_concurrency": 2}},
        [a, b],
    )
    sweep.run(specs_for("pod-a", 3) + specs_for("pod-b", 3, 100), check_clocks=False)
    assert sweep._group_limit("big-box") == 2


def test_the_tightest_member_limit_wins(tmp_path):
    a = ScriptedDevice("pod-a")
    b = ScriptedDevice("pod-b")
    runner, sweep = wire(
        tmp_path,
        {"pod-a": {"resource_group": "box", "group_max_concurrency": 4},
         "pod-b": {"resource_group": "box", "group_max_concurrency": 1}},
        [a, b],
    )
    assert sweep._group_limit("box") == 1


def test_ungrouped_devices_do_not_appear_as_groups(tmp_path):
    device = ScriptedDevice("laptop")
    runner, sweep = wire(tmp_path, {"laptop": {}}, [device])
    result = sweep.run(specs_for("laptop", 2), check_clocks=False)
    assert result.summary["resource_groups"] == {}


def test_a_shared_group_takes_turns_rather_than_letting_one_device_hog_it(tmp_path):
    """Fairness inside a resource group, which is an experimental control here.

    First come first served means whichever worker reacquires the lock after releasing
    it, which is reliably the one that just finished. On a five arm study sharing one
    CPU that put every arm in a different part of the machine's thermal history.
    """
    arms = [ScriptedDevice(name, delay=0.01)
            for name in ("wsl-host", "k8s-cpu2", "k8s-cpu4", "k8s-unlimited")]
    shared = {"resource_group": "laptop-cpu"}
    runner, sweep = wire(tmp_path, {a.device_id: dict(shared) for a in arms}, arms)

    order = []
    lock = threading.Lock()
    original = ScriptedDevice.execute

    def watched(self, task, timeout_s=None):
        if task.get("kind") != "sysinfo":
            with lock:
                order.append(self.device_id)
        return original(self, task, timeout_s)

    ScriptedDevice.execute = watched
    try:
        specs = []
        for index, arm in enumerate(arms):
            specs += specs_for(arm.device_id, 4, index * 100)
        sweep.run(specs, check_clocks=False)
    finally:
        ScriptedDevice.execute = original

    assert len(order) == 16
    # Every group of four consecutive runs should visit all four arms once.
    for start in range(0, 16, 4):
        window = order[start:start + 4]
        assert sorted(window) == sorted(a.device_id for a in arms), (
            f"runs {start}-{start + 3} were {window}, not one per arm")


def test_a_device_with_nothing_queued_does_not_hold_up_the_rotation(tmp_path):
    busy = ScriptedDevice("busy", delay=0.01)
    idle = ScriptedDevice("idle", delay=0.01)
    shared = {"resource_group": "box"}
    runner, sweep = wire(tmp_path, {"busy": dict(shared), "idle": dict(shared)},
                         [busy, idle])
    result = sweep.run(specs_for("busy", 5), check_clocks=False)
    assert len(result.records) == 5
    assert idle.calls == 0


def test_a_down_device_does_not_hold_up_the_rotation(tmp_path):
    healthy = ScriptedDevice("healthy", delay=0.01)
    dead = ScriptedDevice("dead", delay=0.01, fail_after=0, fail_message="gone")
    runner, sweep = wire(
        tmp_path,
        {"healthy": {"resource_group": "box"},
         "dead": {"resource_group": "box", "max_transport_failures": 1,
                  "recovery_after_s": 0.05}},
        [healthy, dead],
        retry=fast_retry(max_attempts=1),
    )
    started = time.time()
    result = sweep.run(specs_for("healthy", 4) + specs_for("dead", 4, 100),
                       check_clocks=False)
    assert time.time() - started < 60
    assert sum(1 for r in result.records if r.ok) == 4


def test_a_device_that_joins_a_group_late_still_gets_a_turn(tmp_path):
    """A failover target is staffed after the rotation was first built."""
    a = ScriptedDevice("box-a", delay=0.01)
    b = ScriptedDevice("box-b", delay=0.01)
    runner, sweep = wire(
        tmp_path,
        {"box-a": {"resource_group": "box"}, "box-b": {"resource_group": "box"}},
        [a, b],
    )
    # Rotation gets built while only one member is known to it.
    sweep._group_order["box"] = ["box-a"]
    result = sweep.run(specs_for("box-a", 2) + specs_for("box-b", 2, 100),
                       check_clocks=False)
    assert len(result.records) == 4
    assert b.successes == 2, "the late joiner must not starve"
    assert "box-b" in sweep._group_order["box"]


def test_an_agent_that_sends_a_null_start_stamp_does_not_crash_the_check(tmp_path):
    device = ScriptedDevice("odd-agent")
    device.execute = lambda task, timeout_s=None: {
        "status": "ok", "agent_started_epoch_s": None, "agent_epoch_s": time.time(),
        "device": {}, "capabilities": {}}
    runner, sweep = wire(tmp_path, {"odd-agent": {}}, [device])
    entry = sweep.check_clocks(["odd-agent"])["odd-agent"]
    assert entry["status"] == "ok"
