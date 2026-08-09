"""Clock and governor control, so frequency is an experiment axis rather than a variable.

Setting these needs root. The intended arrangement is passwordless sudo for the specific
sysfs writes; every function reports whether it succeeded instead of raising, so a run on
an unprivileged machine degrades to "measured at whatever clock the governor chose" and
records that fact rather than aborting.
"""

from __future__ import annotations

import glob
import subprocess
from typing import List, Optional

_GOVERNOR_GLOB = "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"
_SETSPEED_GLOB = "/sys/devices/system/cpu/cpu*/cpufreq/scaling_setspeed"
_AVAILABLE = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies"


def available() -> bool:
    return bool(glob.glob(_GOVERNOR_GLOB))


def _sudo_write(glob_pattern: str, value: str) -> bool:
    """Write one value to every path matching a glob, via a single privileged shell.

    One sudo call for the whole set, not one per CPU: on an 8 core board that is the
    difference between one process spawn and eight, and this runs between measurements.
    """
    script = f"for f in {glob_pattern}; do echo {value} > $f; done"
    try:
        done = subprocess.run(
            ["sudo", "-n", "sh", "-c", script],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def current_governor() -> Optional[str]:
    for path in sorted(glob.glob(_GOVERNOR_GLOB)):
        try:
            with open(path, "r") as fh:
                return fh.read().strip()
        except OSError:
            continue
    return None


def set_governor(governor: str) -> bool:
    return _sudo_write(_GOVERNOR_GLOB, governor)


def set_frequency_khz(khz: int) -> bool:
    """Pin every core to one clock. Requires the userspace governor, so set it first."""
    if not set_governor("userspace"):
        return False
    return _sudo_write(_SETSPEED_GLOB, str(int(khz)))


def available_frequencies_khz() -> Optional[List[int]]:
    try:
        with open(_AVAILABLE, "r") as fh:
            return sorted(int(v) for v in fh.read().split())
    except (OSError, ValueError):
        return None


def restore(governor: str = "schedutil") -> bool:
    """Put the machine back the way it was found. Always call this after a sweep."""
    return set_governor(governor)


def describe() -> dict:
    return {
        "dvfs_read": available(),
        "dvfs_write": available() and _can_sudo(),
    }


def _can_sudo() -> bool:
    try:
        done = subprocess.run(
            ["sudo", "-n", "true"], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0
