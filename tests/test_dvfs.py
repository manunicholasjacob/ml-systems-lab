"""The governor value comes from a config file and ends up in a privileged shell.

`_sudo_write` builds a `sh -c` script and runs it under `sudo -n`. The value it writes
is `governor:` straight out of the user's YAML, so it has to be quoted on the way in.
These tests capture the script instead of running sudo, and one of them runs the script
against a fake sysfs tree to show that a hostile value is written rather than executed.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from mlsyslab.telemetry import dvfs

INJECTION = "ondemand; touch /tmp/mlsyslab_pwned"


def _capture(monkeypatch):
    """Replace subprocess.run inside dvfs and return the list it records calls into."""
    calls = []

    class Done:
        returncode = 0

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return Done()

    monkeypatch.setattr(dvfs.subprocess, "run", fake_run)
    return calls


def test_sudo_write_quotes_the_value(monkeypatch):
    calls = _capture(monkeypatch)
    assert dvfs.set_governor(INJECTION) is True

    argv = calls[0]
    assert argv[:4] == ["sudo", "-n", "sh", "-c"]
    script = argv[4]
    # The whole value is one quoted word, so the semicolon cannot end the echo.
    assert "'ondemand; touch /tmp/mlsyslab_pwned'" in script
    assert "echo ondemand; touch" not in script


def test_sudo_write_leaves_the_glob_expandable(monkeypatch):
    calls = _capture(monkeypatch)
    dvfs.set_governor("powersave")

    script = calls[0][4]
    # Quoting the glob would write to a file literally named "cpu*", so it stays bare.
    assert f"for f in {dvfs._GOVERNOR_GLOB};" in script
    # shlex.quote adds nothing to a word that needs nothing, which keeps the normal
    # script identical to what it was before the quoting change.
    assert "echo powersave >" in script


def test_set_frequency_sets_userspace_first(monkeypatch):
    calls = _capture(monkeypatch)
    assert dvfs.set_frequency_khz(1_800_000) is True

    assert len(calls) == 2
    assert "echo userspace >" in calls[0][4]
    assert "echo 1800000 >" in calls[1][4]
    assert dvfs._SETSPEED_GLOB in calls[1][4]


def test_sudo_write_reports_failure_rather_than_raising(monkeypatch):
    def refuses(argv, **kwargs):
        raise OSError("sudo is not installed")

    monkeypatch.setattr(dvfs.subprocess, "run", refuses)
    assert dvfs.set_governor("schedutil") is False


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX shell")
def test_the_generated_script_writes_the_value_instead_of_running_it(tmp_path):
    """Run the real script against a fake sysfs tree, without sudo."""
    marker = tmp_path / "pwned"
    fake_sysfs = tmp_path / "cpu0"
    fake_sysfs.mkdir()
    target = fake_sysfs / "scaling_governor"
    target.write_text("schedutil\n")

    captured = {}

    def fake_run(argv, **kwargs):
        captured["script"] = argv[4]

        class Done:
            returncode = 0

        return Done()

    # The shell is POSIX even when the test host is not, so give it POSIX paths.
    hostile = f"ondemand; touch {marker.as_posix()}"
    real_run = dvfs.subprocess.run
    dvfs.subprocess.run = fake_run
    try:
        dvfs._sudo_write(f"{tmp_path.as_posix()}/cpu*/scaling_governor", hostile)
    finally:
        dvfs.subprocess.run = real_run

    subprocess.run(["sh", "-c", captured["script"]], check=True, timeout=15)

    assert target.read_text().strip() == hostile
    assert not marker.exists()
