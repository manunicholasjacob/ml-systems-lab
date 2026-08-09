"""Shared fixtures. All hardware-free: recorded outputs stand in for real devices."""

from __future__ import annotations

import json

import pytest

from mlsyslab.backends.base import RunSpec
from mlsyslab.devices.base import Device


# A trimmed but structurally faithful llama-bench -o json output, captured from
# build 10154 on the i7-12700H. Two rows: a prefill test and a decode test.
LLAMA_BENCH_STDOUT = json.dumps([
    {
        "build_commit": "0e4a03622", "build_number": 10154,
        "cpu_info": "12th Gen Intel(R) Core(TM) i7-12700H", "backends": "CPU",
        "model_filename": "C:/llmpc/models/qwen0.5b-q2k.gguf",
        "model_type": "qwen2 1B Q2_K - Medium",
        "model_size": 332659200, "model_n_params": 494032768,
        "n_threads": 4, "n_prompt": 128, "n_gen": 0, "n_depth": 0,
        "avg_ts": 299.461157, "stddev_ts": 2.5,
    },
    {
        "build_commit": "0e4a03622", "build_number": 10154,
        "cpu_info": "12th Gen Intel(R) Core(TM) i7-12700H", "backends": "CPU",
        "model_filename": "C:/llmpc/models/qwen0.5b-q2k.gguf",
        "model_type": "qwen2 1B Q2_K - Medium",
        "model_size": 332659200, "model_n_params": 494032768,
        "n_threads": 4, "n_prompt": 0, "n_gen": 64, "n_depth": 0,
        "avg_ts": 90.078019, "stddev_ts": 0.9,
    },
])


class FakeDevice(Device):
    """A device whose agent is a canned dict, for testing backends and the runner."""

    kind = "local"

    def __init__(self, device_id="fake", config=None, result=None, existing=None):
        super().__init__(device_id, config or {})
        self.result = result or {"status": "ok"}
        self.existing = set(existing or [])
        self.executed_tasks = []

    def execute(self, task, timeout_s=None):
        self.executed_tasks.append(task)
        if isinstance(self.result, Exception):
            raise self.result
        return dict(self.result)

    def resolve(self, path):
        return path

    def exists(self, path):
        return path in self.existing or not self.existing

    def package_root(self):
        return "/fake/root"

    @property
    def python_executable(self):
        return self.config.get("python", "python3")

    def probe(self, refresh=False):
        return {"device": {"device_id": self.device_id, "kind": self.kind,
                           "os": "Linux", "cpu": "Fake CPU"},
                "capabilities": {"temperature": True, "pmic": False}}


@pytest.fixture
def bench_stdout():
    return LLAMA_BENCH_STDOUT


@pytest.fixture
def fake_device():
    return FakeDevice()


@pytest.fixture
def llm_spec():
    return RunSpec(
        experiment="test", device_id="fake", backend="llamacpp", mode="throughput",
        model="qwen0.5b-q2k", model_path="/models/qwen0.5b-q2k.gguf",
        quantization="Q2_K", prompt_tokens=128, output_tokens=64, threads=4,
        repetitions=3,
    )
