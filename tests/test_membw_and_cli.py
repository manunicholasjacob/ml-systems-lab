import json
import subprocess
import sys

import pytest

from mlsyslab.membw import choose_working_set_mb, STABILITY_THRESHOLD
from mlsyslab.cli import build_parser


def test_working_set_respects_small_ram():
    # A 2 GB board must not be handed a 512 MB x2 working set.
    assert choose_working_set_mb(None, ram_bytes=2_000_000_000) <= 256
    assert choose_working_set_mb(None, ram_bytes=32_000_000_000) == 512
    assert choose_working_set_mb(64, ram_bytes=None) == 64
    assert choose_working_set_mb(1, ram_bytes=None) == 64  # floor


def test_membw_measures_something_real():
    numpy = pytest.importorskip("numpy")
    from mlsyslab.membw import measure

    result = measure(working_set_mb=64, thread_counts=[1, 2], reps=2)
    assert result["status"] == "ok"
    # Any machine that can run the test suite moves more than 1 GB/s.
    assert result["peak_read_GBs"] > 1.0
    assert result["best_kernel"] in ("sum", "max", "dot")
    assert 0 < STABILITY_THRESHOLD < 1


def test_membw_module_emits_sentinels():
    out = subprocess.run(
        [sys.executable, "-m", "mlsyslab.membw", "--mb", "64", "--reps", "1"],
        capture_output=True, timeout=300,
    )
    text = out.stdout.decode()
    assert "===MLSYSLAB-MEMBW-BEGIN===" in text
    payload = json.loads(text.split("===MLSYSLAB-MEMBW-BEGIN===")[1]
                         .split("===MLSYSLAB-MEMBW-END===")[0])
    assert payload["status"] in ("ok", "failed")


def test_cli_parser_knows_every_command():
    parser = build_parser()
    for argv in (["probe"], ["run", "x.yaml", "--dry-run"], ["report", "d"],
                 ["membw"], ["compare", "a", "b"]):
        args = parser.parse_args(argv)
        assert callable(args.func)


def test_compare_command_end_to_end(tmp_path, capsys):
    from mlsyslab.cli import main
    from mlsyslab.schema import RunRecord

    def write(directory, decode):
        record = RunRecord(run_id=f"r{decode}", experiment="e")
        record.device.device_id = "d"
        record.backend.name = "llamacpp"
        record.workload.model = "m"
        record.workload.quantization = "Q4_K_M"
        record.workload.prompt_tokens = 128
        record.workload.output_tokens = 64
        record.knobs.threads = 4
        record.metrics.decode_tps = decode
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{record.run_id}.json").write_text(record.to_json(),
                                                         encoding="utf-8")

    write(tmp_path / "a", 10.0)
    write(tmp_path / "b", 30.0)
    code = main(["compare", str(tmp_path / "a"), str(tmp_path / "b")])
    out = capsys.readouterr().out
    assert code == 0
    assert "3.00" in out  # the B/A ratio
