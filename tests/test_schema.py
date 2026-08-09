import json
import os

from mlsyslab.schema import (RunRecord, Metrics, Workload, compute_run_id, from_dict,
                             load_records)


def test_roundtrip_preserves_metrics():
    record = RunRecord(run_id="abc", experiment="x")
    record.metrics.decode_tps = 12.5
    record.metrics.ttft_ms = 340.0
    record.workload.model = "m"
    rebuilt = from_dict(json.loads(record.to_json()))
    assert rebuilt.metrics.decode_tps == 12.5
    assert rebuilt.metrics.ttft_ms == 340.0
    assert rebuilt.workload.model == "m"


def test_to_dict_drops_none_but_keeps_zero():
    record = RunRecord(run_id="abc")
    record.metrics.decode_tps = 0.0
    payload = record.to_dict()
    assert "ttft_ms" not in payload["metrics"]
    # Zero is a measurement, None is an absence; only the absence disappears.
    assert payload["metrics"]["decode_tps"] == 0.0


def test_from_dict_tolerates_unknown_fields():
    payload = {"schema_version": "9.9", "run_id": "abc", "brand_new_field": 1,
               "metrics": {"decode_tps": 5.0, "future_metric": 7.0}}
    record = from_dict(payload)
    assert record.run_id == "abc"
    assert record.metrics.decode_tps == 5.0


def test_run_id_deterministic_and_sensitive():
    a = compute_run_id({"model": "x", "threads": 4})
    b = compute_run_id({"threads": 4, "model": "x"})   # order must not matter
    c = compute_run_id({"model": "x", "threads": 8})
    assert a == b
    assert a != c
    assert len(a) == 12


def test_load_records_ignores_sidecar_jsonl(tmp_path):
    good = RunRecord(run_id="ok1", experiment="e")
    good.metrics.decode_tps = 1.0
    (tmp_path / "run.json").write_text(good.to_json(), encoding="utf-8")
    # The runner's index.jsonl has no schema_version and must not become records.
    (tmp_path / "index.jsonl").write_text(
        json.dumps({"run_id": "ok1", "path": "run.json"}) + "\n", encoding="utf-8"
    )
    records = load_records(str(tmp_path))
    assert len(records) == 1
    assert records[0].run_id == "ok1"


def test_load_records_reads_nested_dirs_and_lists(tmp_path):
    sub = tmp_path / "device-a"
    sub.mkdir()
    one = RunRecord(run_id="r1")
    two = RunRecord(run_id="r2")
    (sub / "r1.json").write_text(one.to_json(), encoding="utf-8")
    (tmp_path / "batch.json").write_text(
        json.dumps([json.loads(two.to_json())]), encoding="utf-8"
    )
    found = {r.run_id for r in load_records(str(tmp_path))}
    assert found == {"r1", "r2"}
