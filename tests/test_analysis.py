import math

import pytest

from mlsyslab.analysis.dataset import (Dataset, decode_roofline,
                                       linear_fit_through_origin)
from mlsyslab.analysis import tables
from mlsyslab.schema import RunRecord


def make_record(device="pi", model="m", weight_bytes=None, decode_tps=None,
                threads=4, quant="Q4_K_M", peak=None, ttft=None, prompt=128,
                latency=None, power=None, stdev=None, throttled=None):
    record = RunRecord(run_id=f"{device}-{model}-{threads}", experiment="t")
    record.device.device_id = device
    record.device.dram_peak_GBs = peak
    record.workload.model = model
    record.workload.weight_bytes = weight_bytes
    record.workload.quantization = quant
    record.workload.prompt_tokens = prompt
    record.knobs.threads = threads
    record.metrics.decode_tps = decode_tps
    record.metrics.ttft_ms = ttft
    record.metrics.latency_ms_mean = latency
    record.metrics.stdev_pct = stdev
    record.telemetry.power_w_mean = power
    record.telemetry.throttled = throttled
    return record


# The paper-12 Pi roofline: decode = 11.34 GB/s / bytes. Synthesise points on that
# exact line and check the fit recovers it.
PI_BW = 11.34e9


def on_the_line(bytes_):
    return PI_BW / bytes_


def test_fit_through_origin_recovers_known_slope():
    xs = [1.0 / b for b in (340e6, 770e6, 940e6)]
    ys = [on_the_line(b) for b in (340e6, 770e6, 940e6)]
    fit = linear_fit_through_origin(xs, ys)
    assert fit["slope"] == pytest.approx(PI_BW)
    assert fit["r2"] == pytest.approx(1.0)


def test_fit_handles_degenerate_input():
    assert math.isnan(linear_fit_through_origin([], [])["slope"])
    assert math.isnan(linear_fit_through_origin([1e-9], [30.0])["slope"])


def test_roofline_uses_best_thread_point_per_model():
    records = [
        make_record(model="a", weight_bytes=340_000_000,
                    decode_tps=on_the_line(340e6), threads=4, peak=13.98),
        # A sub-saturated 1-thread point that would wreck the fit if included.
        make_record(model="a", weight_bytes=340_000_000,
                    decode_tps=on_the_line(340e6) * 0.4, threads=1, peak=13.98),
        make_record(model="b", weight_bytes=770_000_000,
                    decode_tps=on_the_line(770e6), threads=4, peak=13.98),
        make_record(model="c", weight_bytes=940_000_000,
                    decode_tps=on_the_line(940e6), threads=4, peak=13.98),
    ]
    fit = decode_roofline(Dataset(records))
    assert fit["n"] == 3
    assert fit["bw_eff_GBs"] == pytest.approx(11.34, rel=1e-6)
    assert fit["r2"] == pytest.approx(1.0)
    assert fit["peak_fraction_pct"] == pytest.approx(100 * 11.34 / 13.98, rel=1e-6)


def test_dataset_having_excludes_unmeasured_rows():
    records = [make_record(decode_tps=30.0, weight_bytes=340e6),
               make_record(model="latency-only", ttft=500.0)]
    dataset = Dataset(records)
    assert len(dataset.having("decode_tps", "weight_bytes")) == 1
    assert len(dataset.having("ttft_ms")) == 1


def test_integrity_warnings_flag_throttle_and_noise():
    records = [make_record(decode_tps=30.0, stdev=15.0, throttled=True)]
    notes = Dataset(records).integrity_warnings()
    assert any("throttled" in n for n in notes)
    assert any("spread" in n for n in notes)


def test_tables_render_all_three_formats():
    records = [
        make_record(model="a", weight_bytes=340_000_000, decode_tps=33.0, peak=13.98),
        make_record(model="b", weight_bytes=770_000_000, decode_tps=14.7, peak=13.98),
    ]
    dataset = Dataset(records)
    for fmt in ("text", "markdown", "latex"):
        out = tables.roofline_table(dataset, fmt=fmt)
        assert "11" in out or "12" in out  # the fitted GB/s appears
    latex = tables.roofline_table(dataset, fmt="latex")
    assert "\\toprule" in latex and "\\bottomrule" in latex
    markdown = tables.roofline_table(dataset, fmt="markdown")
    assert markdown.count("|") > 10


def test_empty_tables_say_so_instead_of_rendering_headers():
    dataset = Dataset([make_record(latency=2.9)])  # vision only
    assert tables.summary_table(dataset).startswith("(")
    assert tables.latency_table(dataset).startswith("(")
    assert not tables.vision_table(dataset).startswith("(")


def test_flatten_derives_energy_per_token():
    record = make_record(decode_tps=20.0, weight_bytes=340e6, power=4.0)
    row = Dataset([record]).rows[0]
    # 4 W at 20 tok/s is 200 mJ per token.
    assert row["energy_per_token_mj"] == pytest.approx(200.0)
