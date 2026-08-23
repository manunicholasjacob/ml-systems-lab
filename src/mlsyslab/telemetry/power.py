"""Power and energy, from whatever rail the platform exposes.

Two sources are supported, and they are not equivalent:

* **Raspberry Pi PMIC** (``vcgencmd pmic_read_adc``) gives per-rail instantaneous voltage
  and current, so core and DRAM can be separated. It costs a process spawn per sample,
  which is why the default rate is 10 Hz and why throughput must be measured in a
  separate, unsampled run. At 50 Hz the sampler alone slowed decode by roughly 6x.
* **Intel RAPL** (``/sys/class/powercap``) is a monotonic energy counter read from sysfs.
  It is nearly free to sample and gives package energy directly, but it is Linux only and
  says nothing about DRAM on most consumer parts.

* **NVIDIA GPU**, either through NVML's monotonic millijoule counter, which is the RAPL
  shape, or through nvidia-smi for instantaneous watts, which is the PMIC shape. The
  counter is the one to prefer where the part implements it.

Windows exposes neither CPU source without a kernel driver, so it reports None for those.
An absent number is better than an invented one.
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


# ------------------------------------------------------------------ NVIDIA GPU

# Two ways in, and they are not equivalent, which is the same story as PMIC against RAPL.
#
# NVML through `nvidia-ml-py` gives `nvmlDeviceGetTotalEnergyConsumption`, a monotonic
# millijoule counter. That is structurally the RAPL shape: energy comes out directly, the
# read is nearly free, and a GPU run can carry a real energy per token rather than an
# integral of samples. It needs a package installed, which the on-device agent may not
# have, so the import is lazy and its absence is not an error.
#
# `nvidia-smi` is the fallback. It is present wherever a driver is and needs nothing
# installed, but it gives only instantaneous watts and costs a process spawn per sample,
# which is the PMIC shape and carries the same warning: do not sample power in the run you
# take throughput from.
#
# Prefer the counter, and the reason is measured rather than assumed. On an RTX 3050
# (leverage/gpu-access/nvml_rate_sweep_rtx3050.json) the trapezoid of the power samples
# and the counter disagreed by 24 to 41 percent across sampling rates from 0.5 to 20 Hz,
# always in the same direction: the samples read high. The sampler is why. A single
# power read cost 53 to 425 ms, and mean sampled power climbed from 3.4 to 6.7 W as the
# rate rose while the counter said 2.3 to 4.4 W over the same windows, so the act of
# asking was waking the device. Whether that gap survives on datacenter silicon is
# unmeasured, and it is the first thing to check on one.

_SMI = "nvidia-smi"


def _nvml():
    """Import NVML lazily, returning None when it is not installed."""
    try:
        import pynvml  # nvidia-ml-py installs under this name
    except ImportError:
        return None
    return pynvml


def nvml_available() -> bool:
    """True when the energy counter can actually be read on device 0."""
    return read_nvml_energy_mj() is not None


def read_nvml_energy_mj(index: int = 0) -> Optional[int]:
    """Total energy consumed by the GPU since the driver loaded, in millijoules.

    Monotonic, so a delta between two reads is the energy of the window and no
    integration is involved. Returns None when NVML is absent or the part does not
    implement the counter.
    """
    nvml = _nvml()
    if nvml is None:
        return None
    try:
        nvml.nvmlInit()
    except Exception:
        return None
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(index)
        return int(nvml.nvmlDeviceGetTotalEnergyConsumption(handle))
    except Exception:
        return None
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass


def nvml_delta_j(before: int, after: int) -> float:
    """Energy between two counter reads, in joules.

    Unlike RAPL this counter does not wrap in any practical window: it is 64-bit
    millijoules since driver load. A decrease therefore means the driver was reloaded
    between the reads, not a wrap, and the honest answer is zero rather than a guess.
    """
    return (after - before) / 1000.0 if after >= before else 0.0


def gpu_smi_available() -> bool:
    return bool(_run_smi(["--query-gpu=name", "--format=csv,noheader"]))


def _run_smi(args: List[str]) -> Optional[str]:
    try:
        out = subprocess.run([_SMI] + args, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                             timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    text = out.stdout.decode("utf-8", errors="replace").strip()
    return text or None


def read_gpu_power_w(index: int = 0) -> Optional[float]:
    """Instantaneous GPU power draw in watts, via nvidia-smi. One process spawn."""
    text = _run_smi(["-i", str(index), "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits"])
    if not text:
        return None
    try:
        return float(text.splitlines()[0].strip())
    except ValueError:
        return None


def gpu_energy_source(index: int = 0) -> Optional[str]:
    """Which of the two GPU paths this machine can use, if either.

    Named rather than inferred at the call site, because a record that says its energy
    came from a counter and one that says it came from integrated samples are different
    measurements and should not be compared as if they were the same.
    """
    if read_nvml_energy_mj(index) is not None:
        return "nvml_energy_counter"
    if read_gpu_power_w(index) is not None:
        return "nvidia_smi_instantaneous"
    return None


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
    return {
        "pmic": pmic_available(),
        "rapl": rapl_available(),
        "nvml": nvml_available(),
        "nvidia_smi": gpu_smi_available(),
        "gpu_energy_source": gpu_energy_source(),
    }
