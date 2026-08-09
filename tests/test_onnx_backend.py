import json

from mlsyslab.backends.base import RunSpec
from mlsyslab.backends.onnxrt import OnnxRuntimeBackend
from mlsyslab.bench_onnx import RESULT_BEGIN, RESULT_END, _resolve_shape


def make_spec(**overrides):
    base = dict(experiment="test", device_id="fake", backend="onnxruntime",
                mode="throughput", model="cnn-int8", model_path="/models/cnn.onnx",
                quantization="int8", threads=4, repetitions=50)
    base.update(overrides)
    return RunSpec(**base)


def wrap(payload):
    return f"{RESULT_BEGIN}\n{json.dumps(payload)}\n{RESULT_END}\n"


ONNX_PAYLOAD = {
    "status": "ok", "onnxruntime_version": "1.28.0",
    "providers_used": ["CPUExecutionProvider"], "providers_unavailable": [],
    "model": "/models/cnn.onnx",
    "inputs": [{"name": "input", "shape": [1, 3, 32, 32], "type": "tensor(float)"}],
    "threads": 4, "reps": 50, "warmup": 10, "load_ms": 45.0,
    "latency_ms_mean": 2.9, "latency_ms_p50": 2.8, "latency_ms_p95": 3.7,
    "latency_ms_p99": 4.0, "latency_ms_min": 2.6, "latency_ms_max": 4.2,
    "latency_ms_stdev": 0.3, "throughput_ips": 340.0, "wall_s": 0.147,
}


def agent_result(stdout):
    return {"status": "ok", "duration_s": 1.0, "stdout": stdout, "stderr": "",
            "agent_time_utc": "2026-08-09T18:00:00Z",
            "device": {"device_id": "fake", "kind": "local"},
            "telemetry": {}, "warnings": []}


def test_parse_fills_latency_metrics(fake_device):
    record = OnnxRuntimeBackend().parse(make_spec(), agent_result(wrap(ONNX_PAYLOAD)),
                                        fake_device)
    assert record.ok
    assert record.metrics.latency_ms_mean == 2.9
    assert record.metrics.latency_ms_p95 == 3.7
    assert record.metrics.throughput_ips == 340.0
    assert record.backend.version == "1.28.0"
    assert record.workload.input_shape == [1, 3, 32, 32]
    # Token fields must not survive into a vision record looking measured.
    assert record.workload.prompt_tokens is None
    assert record.workload.output_tokens is None


def test_parse_surfaces_missing_providers_as_warning(fake_device):
    payload = dict(ONNX_PAYLOAD)
    payload["providers_unavailable"] = ["CUDAExecutionProvider"]
    record = OnnxRuntimeBackend().parse(make_spec(), agent_result(wrap(payload)),
                                        fake_device)
    assert record.ok
    assert any("CUDAExecutionProvider" in w for w in record.warnings)


def test_parse_failure_payload_fails_the_record(fake_device):
    payload = {"status": "failed", "error": "NoSuchFile: model missing"}
    record = OnnxRuntimeBackend().parse(make_spec(), agent_result(wrap(payload)),
                                        fake_device)
    assert record.status == "failed"
    assert "NoSuchFile" in record.error


def test_parse_garbage_stdout_fails_the_record(fake_device):
    record = OnnxRuntimeBackend().parse(make_spec(),
                                        agent_result("Segmentation fault"), fake_device)
    assert record.status == "failed"


def test_build_task_uses_device_interpreter_and_pythonpath(fake_device):
    fake_device.config["python"] = "/venv/bin/python"
    task = OnnxRuntimeBackend().build_task(make_spec(), fake_device)
    assert task["argv"][0] == "/venv/bin/python"
    assert task["argv"][1:3] == ["-m", "mlsyslab.bench_onnx"]
    assert task["env"]["PYTHONPATH"] == "/fake/root"


def test_resolve_shape_concretises_symbolic_dims():
    assert _resolve_shape([1, 3, 224, 224], batch=1, fallback={}) == [1, 3, 224, 224]
    assert _resolve_shape(["batch", 3, 32, 32], batch=8, fallback={}) == [8, 3, 32, 32]
    assert _resolve_shape(["N", 3, "H", "W"], batch=2,
                          fallback={"H": 64, "W": 48}) == [2, 3, 64, 48]
