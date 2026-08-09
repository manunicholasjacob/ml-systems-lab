import pytest

from mlsyslab.backends.base import RunSpec
from mlsyslab.backends.llamacpp import LlamaCppBackend, _bench_rows, _synthetic_prompt


def make_result(stdout, status="ok", telemetry=None):
    return {
        "status": status, "duration_s": 2.5, "stdout": stdout, "stderr": "",
        "agent_time_utc": "2026-08-09T18:00:00Z",
        "device": {"device_id": "fake", "kind": "local"},
        "telemetry": telemetry or {},
        "warnings": [],
    }


def test_parse_bench_extracts_prefill_and_decode(bench_stdout, llm_spec, fake_device):
    backend = LlamaCppBackend()
    record = backend.parse(llm_spec, make_result(bench_stdout), fake_device)
    assert record.ok
    assert record.metrics.prefill_tps == pytest.approx(299.461157)
    assert record.metrics.decode_tps == pytest.approx(90.078019)
    assert record.workload.weight_bytes == 332659200
    assert record.backend.version == "10154"
    assert record.backend.build == "0e4a03622"
    assert record.workload.params == pytest.approx(0.494, rel=1e-2)


def test_parse_bench_derives_bandwidth_against_declared_peak(bench_stdout, llm_spec):
    from tests.conftest import FakeDevice

    device = FakeDevice(config={"dram_peak_GBs": 53.9})
    record = LlamaCppBackend().parse(llm_spec, make_result(bench_stdout), device)
    expected = 90.078019 * 332659200 / 1e9
    assert record.metrics.decode_bw_GBs == pytest.approx(expected)
    assert record.metrics.bw_utilization_pct == pytest.approx(100 * expected / 53.9)


def test_parse_bench_no_peak_means_no_percentage(bench_stdout, llm_spec, fake_device):
    record = LlamaCppBackend().parse(llm_spec, make_result(bench_stdout), fake_device)
    assert record.metrics.decode_bw_GBs is not None
    assert record.metrics.bw_utilization_pct is None


def test_impossible_utilisation_warns_about_the_ceiling(bench_stdout, llm_spec):
    from tests.conftest import FakeDevice

    device = FakeDevice(config={"dram_peak_GBs": 10.0})  # deliberately too low
    record = LlamaCppBackend().parse(llm_spec, make_result(bench_stdout), device)
    assert record.metrics.bw_utilization_pct > 100
    assert any("declared peak is too low" in w for w in record.warnings)


def test_bench_rows_survive_log_noise(bench_stdout):
    noisy = "ggml_backend_load: loaded CPU backend\n" + bench_stdout + "\nmain: done\n"
    rows = _bench_rows(noisy)
    assert len(rows) == 2


def test_parse_bench_empty_output_fails_the_record(llm_spec, fake_device):
    record = LlamaCppBackend().parse(llm_spec, make_result(""), fake_device)
    assert record.status == "failed"
    assert "no parseable JSON" in record.error


def test_build_bench_task_uses_depth_not_context_flag(llm_spec, fake_device):
    llm_spec.context_tokens = 4096
    task = LlamaCppBackend().build_task(llm_spec, fake_device)
    argv = task["argv"]
    # llama-bench has no -c flag; depth is the only context axis it accepts.
    assert "-c" not in argv
    assert "-d" in argv
    assert argv[argv.index("-d") + 1] == "4096"


def test_build_bench_task_forces_real_prompt_for_decode(llm_spec, fake_device):
    llm_spec.prompt_tokens = 0  # would produce a bogus decode rate
    task = LlamaCppBackend().build_task(llm_spec, fake_device)
    argv = task["argv"]
    assert int(argv[argv.index("-p") + 1]) >= 32


def test_parse_server_latency_medians(llm_spec, fake_device):
    llm_spec.mode = "latency"
    result = make_result("")
    result["load_ms"] = 900.0
    result["requests"] = [
        {"prompt_tokens": 128, "output_tokens": 64, "ttft_ms": 500.0,
         "e2e_latency_ms": 1200.0, "decode_tps": 80.0},
        {"prompt_tokens": 128, "output_tokens": 64, "ttft_ms": 520.0,
         "e2e_latency_ms": 1250.0, "decode_tps": 82.0},
        {"prompt_tokens": 128, "output_tokens": 64, "ttft_ms": 700.0,
         "e2e_latency_ms": 1500.0, "decode_tps": 70.0},
    ]
    record = LlamaCppBackend().parse(llm_spec, result, fake_device)
    assert record.metrics.ttft_ms == 520.0          # median, not mean: robust to one outlier
    assert record.metrics.decode_tps == 80.0
    assert record.metrics.load_ms == 900.0
    assert record.metrics.prefill_tps == pytest.approx(128 / 0.52)


def test_parse_server_all_failed_requests_fails_the_record(llm_spec, fake_device):
    llm_spec.mode = "latency"
    result = make_result("")
    result["requests"] = [{"error": "connection refused"}]
    record = LlamaCppBackend().parse(llm_spec, result, fake_device)
    assert record.status == "failed"


def test_synthetic_prompt_scales_with_target():
    short = _synthetic_prompt(32)
    long = _synthetic_prompt(512)
    assert len(long) > len(short) * 8


def test_telemetry_lands_in_the_record(bench_stdout, llm_spec, fake_device):
    telemetry = {"power_w_mean": 4.2, "temp_c_max": 61.5, "throttled": False,
                 "cpu_util_pct_mean": 87.0, "sample_hz": 10.0}
    record = LlamaCppBackend().parse(
        llm_spec, make_result(bench_stdout, telemetry=telemetry), fake_device
    )
    assert record.telemetry.power_w_mean == 4.2
    assert record.telemetry.temp_c_max == 61.5
    assert record.telemetry.throttled is False
