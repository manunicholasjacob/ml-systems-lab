import json
import sys

import pytest

from mlsyslab.agent import RESULT_BEGIN, RESULT_END, execute
from mlsyslab.devices import from_config
from mlsyslab.devices.base import Device, DeviceError
from mlsyslab.devices.local import LocalDevice


def test_sentinel_parse_survives_ssh_noise():
    payload = {"status": "ok", "value": 42}
    raw = ("Warning: Permanently added host.\nmotd garbage\n"
           + RESULT_BEGIN + "\n" + json.dumps(payload) + "\n" + RESULT_END
           + "\ntrailing logout noise\n")
    assert Device._parse(raw) == payload


def test_sentinel_parse_reports_tail_on_garbage():
    with pytest.raises(DeviceError) as excinfo:
        Device._parse("kernel panic\n" * 3)
    assert "kernel panic" in str(excinfo.value)


def test_agent_unknown_task_fails_cleanly():
    result = execute({"kind": "no-such-task"})
    assert result["status"] == "failed"
    assert "unknown task" in result["error"]
    # Device description still attaches, so even failures are attributable.
    assert "device" in result and "capabilities" in result


def test_agent_command_task_runs_and_samples():
    result = execute({
        "kind": "command",
        "argv": [sys.executable, "-c", "print('out'); import sys; sys.exit(0)"],
        "sample_power": False, "sample_hz": 20,
    })
    assert result["status"] == "ok"
    assert "out" in result["stdout"]
    assert result["telemetry"]["sample_count"] >= 0
    assert result["duration_s"] > 0


def test_agent_command_nonzero_exit_is_failed_but_complete():
    result = execute({
        "kind": "command",
        "argv": [sys.executable, "-c", "import sys; sys.stderr.write('bad'); sys.exit(3)"],
        "sample_power": False,
    })
    assert result["status"] == "failed"
    assert "exit code 3" in result["error"]
    assert "bad" in result["stderr"]


def test_agent_missing_binary_is_failed_not_crash():
    result = execute({"kind": "command", "argv": ["definitely-not-a-binary-xyz"],
                      "sample_power": False})
    assert result["status"] == "failed"


def test_local_device_full_round_trip():
    device = LocalDevice("me", {"python": sys.executable})
    result = device.execute({
        "kind": "command",
        "argv": [sys.executable, "-c", "print(6*7)"],
        "sample_power": False,
    })
    assert result["status"] == "ok"
    assert "42" in result["stdout"]


def test_registry_defaults_ssh_when_host_present():
    device = from_config("pi", {"host": "203.0.113.5", "user": "u"})
    assert device.kind == "ssh"
    local = from_config("here", {})
    assert local.kind == "local"


def test_registry_rejects_unknown_kind():
    with pytest.raises(DeviceError, match="unknown device kind"):
        from_config("x", {"kind": "quantum"})


def test_ssh_device_requires_host():
    with pytest.raises(DeviceError, match="no host"):
        from_config("x", {"kind": "ssh"})
