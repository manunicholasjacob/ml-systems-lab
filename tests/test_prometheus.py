"""The exporter, the dashboard that reads it, and the rule that neither may be the data."""

import json
import os
import pathlib
import re
import urllib.request

import pytest

from mlsyslab.prometheus import (Exporter, Registry, chain, from_directory, _escape,
                                 _format, _labels_for)
from mlsyslab.schema import (BackendInfo, DeviceInfo, Knobs, Metrics, RunRecord,
                             Telemetry, Workload)

REPO = pathlib.Path(__file__).resolve().parents[1]


def make_record(device="pi5", model="qwen0.5b", quant="Q4_K_M", decode=42.5,
                status="ok", ttft=None, power=5.5, temp=61.0, threads=4):
    return RunRecord(
        run_id="abc123", experiment="test", status=status,
        device=DeviceInfo(device_id=device, kind="ssh"),
        backend=BackendInfo(name="llamacpp"),
        workload=Workload(model=model, quantization=quant, prompt_tokens=128,
                          output_tokens=64),
        knobs=Knobs(threads=threads),
        metrics=Metrics(decode_tps=decode, prefill_tps=300.0, ttft_ms=ttft,
                        peak_rss_bytes=1234567, bw_utilization_pct=87.0),
        telemetry=Telemetry(power_w_mean=power, temp_c_max=temp,
                            energy_per_token_mj=120.0, throttled=False),
    )


# ----------------------------------------------------------------- exposition format

def test_render_is_valid_exposition_text():
    registry = Registry(experiment="demo")
    registry.observe_record(make_record())
    text = registry.render()

    assert text.endswith("\n")
    for line in text.splitlines():
        if line.startswith("#"):
            assert re.match(r"^# (HELP|TYPE) mlsyslab_[a-zA-Z_:][a-zA-Z0-9_:]* ", line)
        else:
            assert re.match(r"^mlsyslab_[a-zA-Z0-9_:]+(\{[^}]*\})? [^ ]+$", line), line

    # Every series is preceded by a declared HELP and TYPE.
    names = {line.split("{")[0].split(" ")[0]
             for line in text.splitlines() if not line.startswith("#")}
    for name in names:
        assert f"# TYPE {name} " in text
        assert f"# HELP {name} " in text


def test_metric_values_land_where_expected():
    registry = Registry(experiment="demo")
    registry.observe_record(make_record(decode=42.5))
    labels = dict(device="pi5", backend="llamacpp", model="qwen0.5b", quant="Q4_K_M",
                  mode="throughput", threads=4, tokens="p128n64", experiment="demo")
    assert registry.get("decode_tokens_per_second", **labels) == 42.5
    assert registry.get("power_watts", **labels) == 5.5
    assert registry.get("temperature_celsius", **labels) == 61.0
    # A percentage is exported as a ratio, because that is what the unit says.
    assert registry.get("bandwidth_utilization_ratio", **labels) == pytest.approx(0.87)


def test_label_escaping_survives_a_hostile_value():
    assert _escape('a"b\\c\nd') == 'a\\"b\\\\c\\nd'
    registry = Registry(experiment='weird"name')
    registry.observe_record(make_record(model='m\\1'))
    text = registry.render()
    assert 'experiment="weird\\"name"' in text
    assert 'model="m\\\\1"' in text


def test_numbers_render_without_scientific_surprises():
    assert _format(1.0) == "1"
    assert _format(0.5) == "0.5"
    assert _format(float("inf")) == "+Inf"
    assert _format(float("nan")) == "NaN"


# --------------------------------------------------------------------- label design

def test_mode_is_derived_the_same_way_the_analysis_layer_derives_it():
    assert _labels_for(make_record(ttft=None))["mode"] == "throughput"
    assert _labels_for(make_record(ttft=88.0))["mode"] == "latency"


def test_every_swept_axis_is_a_label_so_points_do_not_collapse():
    """Two points that differ only in threads must be two series, not one."""
    registry = Registry(experiment="demo")
    registry.observe_record(make_record(threads=1, decode=10.0))
    registry.observe_record(make_record(threads=4, decode=40.0))
    text = registry.render()
    assert 'threads="1"' in text and 'threads="4"' in text
    assert "10" in text and "40" in text


def test_run_ids_never_become_labels():
    """The unbounded ones stay in the records, where they belong."""
    registry = Registry(experiment="demo")
    registry.observe_record(make_record())
    text = registry.render()
    assert "abc123" not in text
    assert "run_id" not in text


def test_a_failed_run_is_counted_but_publishes_no_measurements():
    registry = Registry(experiment="demo")
    registry.observe_record(make_record(status="failed"))
    assert registry.get("sweep_runs_total", device="pi5", status="failed",
                        experiment="demo") == 1
    # Not even a declaration: an empty series would read as "measured, and it was zero".
    assert "decode_tokens_per_second" not in registry.render()


# ------------------------------------------------------------- observing a live sweep

def test_observer_tracks_devices_going_down_and_coming_back():
    registry = Registry(experiment="demo")
    observe = registry.observer()

    observe("clock", {"device": "pi5", "status": "ok", "offset_s": 0.42})
    assert registry.get("device_up", device="pi5", experiment="demo") == 1
    assert registry.get("device_clock_offset_seconds", device="pi5",
                        experiment="demo") == 0.42

    observe("device_down", {"device": "pi5", "reason": "ssh died",
                            "consecutive_failures": 3})
    assert registry.get("device_up", device="pi5", experiment="demo") == 0
    assert registry.get("device_transport_failures_total", device="pi5",
                        experiment="demo") == 3

    observe("device_up", {"device": "pi5", "recoveries": 1})
    assert registry.get("device_up", device="pi5", experiment="demo") == 1


def test_observer_balances_inflight_across_start_and_finish():
    registry = Registry(experiment="demo")
    observe = registry.observer()

    class Spec:
        device_id = "laptop"

    observe("start", {"spec": Spec()})
    observe("start", {"spec": Spec()})
    assert registry.get("sweep_runs_inflight", device="laptop", experiment="demo") == 2
    observe("finish", {"spec": Spec(), "record": make_record(device="laptop")})
    assert registry.get("sweep_runs_inflight", device="laptop", experiment="demo") == 1


def test_watching_never_breaks_running():
    """A registry that raises would take a sweep down with it. It must not."""
    registry = Registry(experiment="demo")
    observe = registry.observer()
    observe("finish", {"spec": None, "record": object()})    # nonsense, on purpose
    registry.observe_record(None)
    assert registry.render()                                  # still serviceable


def test_chain_fans_events_out_and_tolerates_a_missing_listener():
    seen = []
    fan = chain(lambda k, p: seen.append(("a", k)), None,
                lambda k, p: seen.append(("b", k)))
    fan("finish", {})
    assert seen == [("a", "finish"), ("b", "finish")]


# ------------------------------------------------------------- reading from records

def test_from_directory_projects_records_that_are_already_on_disk(tmp_path):
    (tmp_path / "pi5").mkdir()
    for index, decode in enumerate((10.0, 20.0)):
        record = make_record(decode=decode, threads=index + 1)
        (tmp_path / "pi5" / f"r{index}.json").write_text(record.to_json(),
                                                        encoding="utf-8")
    registry = from_directory(str(tmp_path))
    text = registry.render()
    assert "mlsyslab_decode_tokens_per_second" in text
    assert "10" in text and "20" in text


def test_from_directory_ignores_the_runners_own_sidecars(tmp_path):
    (tmp_path / "index.jsonl").write_text('{"run_id": "x", "status": "ok"}\n',
                                          encoding="utf-8")
    (tmp_path / "schedule.json").write_text('{"mode": "concurrent"}', encoding="utf-8")
    registry = from_directory(str(tmp_path))
    assert "sweep_runs_total" not in registry.render()


# --------------------------------------------------------------------------- serving

def test_metrics_endpoint_serves_the_registry():
    registry = Registry(experiment="demo")
    registry.observe_record(make_record())
    with Exporter(registry, port=0) as exporter:
        with urllib.request.urlopen(exporter.url, timeout=10) as response:
            assert response.status == 200
            assert "text/plain" in response.headers["Content-Type"]
            body = response.read().decode("utf-8")
    assert "mlsyslab_decode_tokens_per_second" in body


def test_anything_other_than_metrics_is_a_404():
    with Exporter(Registry(), port=0) as exporter:
        url = exporter.url.replace("/metrics", "/admin")
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(url, timeout=10)
        assert excinfo.value.code == 404


def test_exporter_binds_to_loopback_unless_told_otherwise():
    exporter = Exporter(Registry(), port=0)
    assert exporter.host == "127.0.0.1"


# --------------------------------------------------------------------- the dashboard

DASHBOARD = REPO / "grafana" / "mlsyslab-sweep.json"


def test_dashboard_is_valid_json_and_checked_in():
    assert DASHBOARD.exists()
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    assert dashboard["uid"] == "mlsyslab-sweep"
    assert dashboard["panels"]


def test_every_dashboard_query_names_a_metric_the_exporter_actually_exports():
    """A dashboard panel that queries a metric nobody publishes is a blank panel."""
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    exported = set(re.findall(r"^# TYPE (mlsyslab_\w+) ",
                              _fully_populated_registry().render(), re.MULTILINE))

    queried = set()
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []) or []:
            queried.update(re.findall(r"mlsyslab_\w+", target.get("expr", "")))

    missing = queried - exported
    assert not missing, f"dashboard queries metrics nothing exports: {sorted(missing)}"


def test_dashboard_says_out_loud_that_it_is_not_the_data():
    text = DASHBOARD.read_text(encoding="utf-8")
    assert "run records" in text
    assert "never from here" in text


def _fully_populated_registry() -> Registry:
    """A registry with one of everything, for checking the dashboard against."""
    registry = Registry(experiment="demo")
    registry.observe_record(make_record(ttft=90.0))
    registry.observe_record(make_record(status="failed"))
    observe = registry.observer()
    observe("clock", {"device": "pi5", "status": "ok", "offset_s": 0.1})
    observe("device_down", {"device": "pi5", "consecutive_failures": 1})
    observe("retry", {"spec": type("S", (), {"device_id": "pi5"})()})
    registry.set("device_cpu_quota_cores", 4.0, device="k8s-cpu")
    registry.observe_queue({"pi5": 3})
    registry.set("sweep_runs_inflight", 1.0, device="pi5")
    registry.observe_record(make_record(device="laptop", ttft=None))
    return registry


def test_observe_devices_records_only_what_a_device_can_actually_report():
    class Quota:
        def cgroup_limits(self):
            return {"cpu_quota_cores": 2.0}

    class Plain:
        pass

    registry = Registry(experiment="demo")
    registry.observe_devices({"pod": Quota(), "laptop": Plain()})
    assert registry.get("device_cpu_quota_cores", device="pod", experiment="demo") == 2.0
    assert registry.get("device_cpu_quota_cores", device="laptop",
                        experiment="demo") is None
