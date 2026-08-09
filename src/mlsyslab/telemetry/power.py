"""Power and energy, from whatever rail the platform exposes.

Two sources are supported, and they are not equivalent:

* **Raspberry Pi PMIC** (``vcgencmd pmic_read_adc``) gives per-rail instantaneous voltage
  and current, so core and DRAM can be separated. It costs a process spawn per sample,
  which is why the default rate is 10 Hz and why throughput must be measured in a
  separate, unsampled run. At 50 Hz the sampler alone slowed decode by roughly 6x.
* **Intel RAPL** (``/sys/class/powercap``) is a monotonic energy counter read from sysfs.
  It is nearly free to sample and gives package energy directly, but it is Linux only and
  says nothing about DRAM on most consumer parts.

Windows exposes neither without a kernel driver, so it reports None. An absent number is
better than an invented one.
"""

from __future__ import annotations

import glob
import os
import subprocess
import time
from typing import Dict, List, Optional, Tuple

_PMIC = "/usr/bin/vcgencmd"
_RAPL_GLOB = "/sys/class/powercap/intel-rapl:*/energy_uj"


# ---------------------------------------------------------------- Raspberry Pi PMIC

def pmic_available() -> bool:
    return os.path.exists(_PMIC)


def read_pmic() -> Optional[Tuple[float, float, float]]:
    """Instantaneous (total_w, core_w, dram_w) across every PMIC rail.

    Power is summed as V*I per rail. The core rails are the ones that dominate; the DRAM
    rails typically come to only 4 to 5 percent of the total on a Pi 5, which is the
    measurement that says edge inference energy is core-stall energy rather than data
    movement energy.
    """
    try:
        out = subprocess.run(
            [_PMIC, "pmic_read_adc"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None

    volts: Dict[str, float] = {}
    amps: Dict[str, float] = {}
    for line in out.stdout.decode("utf-8", "replace").strip().splitlines():
        line = line.strip()
        if "=" not in line or " " not in line:
            continue
        label, value = line.split()[0], line.split("=")[-1]
        try:
            number = float(value.rstrip("VA"))
        except ValueError:
            continue
        if "volt(" in line:
            volts[label[:-2]] = number
        elif "current(" in line:
            amps[label[:-2]] = number

    total = core = dram = 0.0
    for rail, v in volts.items():
        if rail not in amps:
            continue
        watts = v * amps[rail]
        total += watts
        if rail.startswith("VDD_CORE"):
            core += watts
        elif rail.startswith("DDR"):
            dram += watts
    return (total, core, dram) if total else None


# -------------------------------------------------------------------- Intel RAPL

def rapl_available() -> bool:
    return bool(glob.glob(_RAPL_GLOB))


def read_rapl_uj() -> Optional[int]:
    """Sum of every RAPL domain's energy counter, in microjoules.

    The counter wraps. Callers must treat a decrease as a wrap rather than as negative
    energy, which is what :func:`rapl_delta_j` does.
    """
    total = 0
    found = False
    for path in sorted(glob.glob(_RAPL_GLOB)):
        try:
            with open(path, "r") as fh:
                total += int(fh.read().strip())
            found = True
        except (OSError, ValueError, PermissionError):
            continue
    return total if found else None


def rapl_max_uj() -> Optional[int]:
    for path in sorted(glob.glob(_RAPL_GLOB)):
        cap = path.replace("energy_uj", "max_energy_range_uj")
        try:
            with open(cap, "r") as fh:
                return int(fh.read().strip())
        except (OSError, ValueError):
            continue
    return None


def rapl_delta_j(before: int, after: int) -> float:
    """Energy between two counter reads, correcting for a single wrap."""
    if after >= before:
        return (after - before) / 1e6
    ceiling = rapl_max_uj()
    if ceiling:
        return ((ceiling - before) + after) / 1e6
    return 0.0


# ------------------------------------------------------------------- integration

def integrate_energy(samples: List[Tuple[float, float]]) -> Optional[float]:
    """Trapezoidal integral of (timestamp_s, watts) pairs, in joules.

    Trapezoidal rather than rectangular because the sample rate is deliberately low to
    stay out of the way, and at 10 Hz the rectangular error on a ramping workload is not
    negligible.
    """
    if len(samples) < 2:
        return None
    energy = 0.0
    for i in range(1, len(samples)):
        dt = samples[i][0] - samples[i - 1][0]
        if dt <= 0:
            continue
        energy += 0.5 * (samples[i][1] + samples[i - 1][1]) * dt
    return energy


def measure_idle_power_w(seconds: float = 3.0, period: float = 0.1) -> Optional[float]:
    """Baseline power with nothing running, needed to report marginal inference energy."""
    if not pmic_available():
        return None
    readings = []
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        reading = read_pmic()
        if reading:
            readings.append(reading[0])
        time.sleep(period)
    return sum(readings) / len(readings) if readings else None


def describe() -> dict:
    return {"pmic": pmic_available(), "rapl": rapl_available()}
