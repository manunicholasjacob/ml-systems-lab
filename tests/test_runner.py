import json
import os

import pytest

from mlsyslab import backends as backend_registry
from mlsyslab.backends.base import Backend, RunSpec
from mlsyslab.config import Config
from mlsyslab.runner import Runner
from mlsyslab.schema import BackendInfo, RunRecord, compute_run_id

from tests.conftest import FakeDevice, LLAMA_BENCH_STDOUT


class CountingBackend(Backend):
    """Delegates to the real llama.cpp parser but counts invocations."""

    name = "counting"

    def __init__(self, config=None):
        super().__init__(config)
        self.tasks_built = 0

    def discover(self, device):
        return BackendInfo(name=self.name)

    def build_task(self, spec, device):
        self.tasks_built += 1
        return {"kind": "command", "argv": ["fake"]}

    def parse(self, spec, result, device):
        from mlsyslab.backends.llamacpp import LlamaCppBackend

        return LlamaCppBackend().parse(spec, result, device)


@pytest.fixture
def wired_runner(tmp_path, monkeypatch):
    backend_registry.register("counting", CountingBackend)
    config = Config(
        experiment="test", output_dir=str(tmp_path / "out"),
        devices={"fake": {"kind": "local"}},
    )
    good_result = {
        "status": "ok", "duration_s": 1.0, "stdout": LLAMA_BENCH_STDOUT, "stderr": "",
        "agent_time_utc": "2026-08-09T18:00:00Z",
        "device": {"device_id": "fake", "kind": "local"}, "telemetry": {},
        "warnings": [],
    }
    runner = Runner(config, resume=True)
    device = FakeDevice(result=good_result)
    runner._devices["fake"] = device
    return runner, device


def make_spec(threads=4):
    return RunSpec(experiment="test", device_id="fake", backend="counting",
                   model="m", model_path="/m.gguf", threads=threads,
                   prompt_tokens=128, output_tokens=64)


def test_run_writes_record_and_index(wired_runner):
    runner, _ = wired_runner
    records = runner.run([make_spec()])
    assert len(records) == 1
    assert records[0].ok
    path = runner.record_path(make_spec())
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["metrics"]["decode_tps"] == pytest.approx(90.078019)
    index = os.path.join(runner.output_dir, "index.jsonl")
    assert os.path.exists(index)
    manifest = os.path.join(runner.output_dir, "manifest.json")
    assert os.path.exists(manifest)


def test_resume_skips_completed_points(wired_runner):
    runner, device = wired_runner
    runner.run([make_spec()])
    executed_before = len(device.executed_tasks)
    records = runner.run([make_spec()])   # same spec again
    assert len(records) == 0              # nothing executed
    assert len(device.executed_tasks) == executed_before


def test_resume_retries_failed_points(wired_runner):
    runner, device = wired_runner
    good = dict(device.result)
    device.result = {"status": "failed", "error": "boom", "stdout": "", "stderr": "",
                     "device": {}, "telemetry": {}, "warnings": []}
    records = runner.run([make_spec()])
    assert not records[0].ok
    device.result = good
    records = runner.run([make_spec()])
    assert len(records) == 1              # the failed point ran again
    assert records[0].ok


def test_device_error_becomes_failed_record(wired_runner):
    runner, device = wired_runner
    device.result = RuntimeError("ssh died")
    records = runner.run([make_spec(threads=8)])
    assert len(records) == 1
    assert records[0].status == "failed"
    assert "ssh died" in records[0].error
    # And the failure is on disk, not just in memory.
    with open(runner.record_path(make_spec(threads=8)), encoding="utf-8") as fh:
        assert json.load(fh)["status"] == "failed"


def test_no_partial_files_left_behind(wired_runner):
    runner, _ = wired_runner
    runner.run([make_spec()])
    leftovers = [name for _, _, names in os.walk(runner.output_dir)
                 for name in names if name.endswith(".partial")]
    assert leftovers == []
