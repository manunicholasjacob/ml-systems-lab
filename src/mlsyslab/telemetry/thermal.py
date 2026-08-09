"""Temperature and throttle state.

Read from sysfs wherever possible. ``vcgencmd measure_temp`` gives the same number on a
Pi but costs a process spawn per sample, and a sampler that spawns processes at 10 Hz is
competing with the thing it is measuring. sysfs is a file read.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
from typing import List, Optional, Tuple

_ZONE_GLOB = "/sys/class/thermal/thermal_zone*/temp"


def _zone_paths() -> List[str]:
    return sorted(glob.glob(_ZONE_GLOB))


def available() -> bool:
    return bool(_zone_paths())


def read_temp_c() -> Optional[float]:
    """Hottest thermal zone in Celsius, or None if the platform exposes none.

    Windows has no equivalent that works without administrator rights or a third-party
    driver, so it reports None rather than a number that might be wrong.
    """
    temps = []
    for path in _zone_paths():
        try:
            with open(path, "r") as fh:
                raw = fh.read().strip()
        except OSError:
            continue
        if not raw.lstrip("-").isdigit():
            continue
        value = int(raw)
        # Zones report millidegrees; a few report degrees. Anything above 200 is
        # millidegrees, since no CPU this framework targets runs at 200 C.
        temps.append(value / 1000.0 if abs(value) > 200 else float(value))
    return max(temps) if temps else None


def read_throttle_flags() -> Tuple[Optional[bool], Optional[str]]:
    """Raspberry Pi throttle bitmask, as (currently_throttled, raw_hex).

    Bit 0 is under-voltage now, bit 1 is frequency capped now, bit 2 is throttled now,
    bit 3 is soft temperature limit now. Bits 16 and up are sticky "has happened since
    boot" versions, which are not evidence about this run and are excluded.
    """
    try:
        out = subprocess.run(
            ["vcgencmd", "get_throttled"], stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    if out.returncode != 0:
        return None, None
    m = re.search(r"throttled=(0x[0-9a-fA-F]+)", out.stdout.decode("utf-8", "replace"))
    if not m:
        return None, None
    raw = m.group(1)
    return bool(int(raw, 16) & 0xF), raw


def cpu_freq_khz() -> Optional[List[int]]:
    """Current clock of every CPU, in kHz."""
    freqs = []
    for path in sorted(glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq")):
        try:
            with open(path, "r") as fh:
                freqs.append(int(fh.read().strip()))
        except (OSError, ValueError):
            continue
    if freqs:
        return freqs
    # Some x86 kernels expose the current clock only via /proc/cpuinfo MHz.
    try:
        with open("/proc/cpuinfo", "r", errors="replace") as fh:
            mhz = re.findall(r"^cpu MHz\s*:\s*([\d.]+)$", fh.read(), re.MULTILINE)
    except OSError:
        return None
    return [int(float(v) * 1000) for v in mhz] or None


def describe() -> dict:
    """What this platform can actually report, for the capability model."""
    return {
        "temperature": available(),
        "throttle_flags": os.path.exists("/usr/bin/vcgencmd"),
        "cpu_freq": bool(glob.glob("/sys/devices/system/cpu/cpu*/cpufreq")),
    }
