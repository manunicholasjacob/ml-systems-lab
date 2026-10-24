"""Integrity checks for the recorded OpenVINO ``benchmark_app`` output (issue #3).

There is no OpenVINO parser yet; these files exist so that one can be written and
tested against real output instead of output typed from memory. What is checked here is
only that the recordings stay what was recorded: every file is listed in the manifest,
its bytes match the recorded SHA-256, and the manifest says how each one was produced.
A parser test should read the files through ``openvino_fixture()`` below.
"""

import hashlib
import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "recorded" / "openvino"
MANIFEST = json.loads((FIXTURES / "MANIFEST.json").read_text(encoding="utf-8"))
SKIP = {"MANIFEST.json", "README.md"}


def openvino_fixture(name: str) -> str:
    """Return a recorded stream as text, exactly as benchmark_app wrote it."""
    return (FIXTURES / name).read_bytes().decode("utf-8")


def test_every_recorded_file_is_in_the_manifest():
    on_disk = {p.name for p in FIXTURES.iterdir() if p.is_file() and p.name not in SKIP}
    listed = set()
    for run in MANIFEST["runs"]:
        listed.update(run["files"].values())
    assert on_disk == listed


@pytest.mark.parametrize("run", MANIFEST["runs"], ids=lambda r: r["id"])
def test_recorded_bytes_are_unedited(run):
    for name in run["files"].values():
        digest = hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()
        assert digest == MANIFEST["sha256"][name], f"{name} differs from the recording"


@pytest.mark.parametrize("run", MANIFEST["runs"], ids=lambda r: r["id"])
def test_each_run_says_how_it_was_produced(run):
    assert "benchmark_app" in run["argv"][0]
    assert isinstance(run["exit_code"], int)
    assert run["device"] in ("CPU", "GPU.0")
    if run["exit_code"] == 0:
        assert "Throughput" in openvino_fixture(run["files"]["stdout"])
