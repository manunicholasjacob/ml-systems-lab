"""Power and energy, from whatever rail the platform exposes.

Two sources are supported, and they are not equivalent:

* **Raspberry Pi PMIC** (``vcgencmd pmic_read_adc``) gives per-rail instantaneous voltage
  and current, so core and DRAM can be separated. It costs a process spawn per sample,
  which is why the default rate is 10 Hz and why throughput must be measured in a
  separate, unsampled run. At 50 Hz the sampler alone slowed decode by roughly 6x.
* **Intel RAPL** (``/sys/class/powercap``) is a monotonic energy counter read from sysfs.
  It is nearly free to sample and gives package energy directly, but it is Linux only and
  says nothing about DRAM on most consumer parts.

* **NVIDIA GPU**, through one of three instruments depending on what the part supports:
  NVML's monotonic millijoule counter (the RAPL shape, Volta and newer), integrated NVML
  power samples, or integrated nvidia-smi samples (the PMIC shape, one process spawn
  per reading). Which one produced a number is recorded with it, because they are not
  interchangeable.

Windows exposes neither CPU source without a kernel driver, so it reports None for those.
An absent number is better than an invented one.
"""

from __future__ import annotations

import glob
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

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
    """Which instrument this device can actually give energy from, if any.

    Named rather than inferred at the call site, because a monotonic counter and a
    trapezoid of samples are different measurements and should not be compared as if they
    were the same. See ENERGY_SOURCES below for why there are three rungs rather than two.
    """
    if read_nvml_energy_mj(index) is not None:
        return "nvml_energy_counter"
    if read_nvml_power_w(index) is not None:
        return "nvml_integrated_samples"
    if read_gpu_power_w(index) is not None:
        return "nvidia_smi_instantaneous"
    return None


# The instrument ladder, and why it has three rungs rather than two.
#
# Round one measured NVML on an RTX 3050 Laptop and concluded: prefer the counter, distrust
# the sampler. Sampling at 20 Hz raised the card's own energy counter from 0.233 W unpolled
# to 4.297 W, 18 times higher, and each power read blocked for 53 to 425 ms.
#
# The same probe on a Tesla P100 says the opposite about the sampler and removes the
# counter. Calls cost 3 ms and are flat with rate; the sampled mean moves 0.1 W across a
# twentyfold change in sampling rate. And `nvmlDeviceGetTotalEnergyConsumption` raises
# NVMLError_NotSupported, because NVIDIA scopes that counter to Volta and newer and the
# P100 is Pascal.
#
# So neither single-device conclusion was the rule. The rule is that the right primitive
# depends on the part and you have to ask it:
#
#   consumer laptop GPU   counter exists, sampler lies      -> counter
#   Pascal datacenter     counter absent, sampler is cheap  -> integrate samples
#   Volta and newer       both exist                        -> counter, and cross-check
#
# Every energy figure records which rung produced it. Two devices' numbers are otherwise
# silently incomparable, which is the failure the framework's absent-never-zero rule exists
# to prevent.
#
# Evidence: leverage/gpu-access/nvml_rate_sweep_rtx3050.json,
# nvml_unpolled_baseline_rtx3050.json, and the Kaggle P100 run in NVML_CROSSCHECK.md.

ENERGY_SOURCES = (
    "nvml_energy_counter",        # monotonic mJ, Volta and newer
    "nvml_integrated_samples",    # trapezoid of nvmlDeviceGetPowerUsage, ~3 ms per read
    "nvidia_smi_instantaneous",   # trapezoid of nvidia-smi, one process spawn per sample
)


def read_nvml_power_w(index: int = 0) -> Optional[float]:
    """Instantaneous GPU power through NVML, in watts.

    Two orders of magnitude cheaper than spawning nvidia-smi, which is what makes
    integrating samples viable on a part with no energy counter.
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
        return nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    except Exception:
        return None
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass


def gpu_capabilities(index: int = 0) -> Dict[str, Any]:
    """What this particular GPU will answer, measured rather than assumed.

    Every getter is called individually and an unsupported one reports absent instead of
    raising. Prong 5 lost a Kaggle pre-flight cell to a single unguarded call taking the
    whole notebook down with it.
    """
    caps: Dict[str, Any] = {
        "nvml_installed": _nvml() is not None,
        "energy_counter": read_nvml_energy_mj(index) is not None,
        "nvml_power": read_nvml_power_w(index) is not None,
        "nvidia_smi_power": read_gpu_power_w(index) is not None,
    }
    caps["energy_source"] = gpu_energy_source(index)
    caps["can_measure_energy"] = caps["energy_source"] is not None
    return caps


def sample_gpu_power(seconds: float, hz: float = 10.0,
                     index: int = 0) -> Optional[List[Tuple[float, float]]]:
    """(timestamp, watts) pairs from the cheapest instantaneous source available.

    For a part with no energy counter this is the measurement, so it uses NVML when it is
    there and falls back to nvidia-smi only when it is not. On a consumer card the act of
    sampling moves the number; on a datacenter card it does not. The caller decides whether
    that matters by reading which source :func:`gpu_energy_source` named.
    """
    # Resolve the reader once. On a part where sampling perturbs the device, every extra
    # read is a measurement of the measurement.
    source = gpu_energy_source(index)
    if source == "nvidia_smi_instantaneous":
        reader = read_gpu_power_w
    elif source in ("nvml_integrated_samples", "nvml_energy_counter"):
        reader = read_nvml_power_w
    else:
        return None
    out: List[Tuple[float, float]] = []
    interval = 1.0 / hz if hz > 0 else 0.1
    start = time.perf_counter()
    deadline = start + seconds
    while time.perf_counter() < deadline:
        watts = reader(index)
        if watts is not None:
            out.append((time.perf_counter() - start, watts))
        time.sleep(interval)
    return out or None


def gpu_energy_j(seconds: float, hz: float = 10.0,
                 index: int = 0) -> Optional[Dict[str, Any]]:
    """Energy over a window, by whichever instrument this device actually has.

    Returns the joules, the source that produced them, and on a part that has both, the
    ratio between them. That cross-check costs a minute and is worth it: on the RTX 3050
    the integral overstated the counter by 24 to 41 percent.
    """
    before = read_nvml_energy_mj(index)
    samples = sample_gpu_power(seconds, hz, index)
    after = read_nvml_energy_mj(index)

    out: Dict[str, Any] = {"window_s": seconds, "sample_hz": hz,
                           "n_samples": len(samples) if samples else 0}
    integrated = integrate_energy(samples) if samples else None
    if integrated is not None:
        out["energy_j_integrated"] = round(integrated, 4)
    if before is not None and after is not None:
        out["energy_j_counter"] = round(nvml_delta_j(before, after), 4)

    if "energy_j_counter" in out:
        out["energy_j"] = out["energy_j_counter"]
        out["energy_source"] = "nvml_energy_counter"
        if integrated:
            out["counter_over_integrated"] = round(out["energy_j_counter"] / integrated, 4)
    elif integrated is not None:
        out["energy_j"] = out["energy_j_integrated"]
        source = gpu_energy_source(index)
        out["energy_source"] = source
        out["note"] = (
            "no energy counter available, so energy is the trapezoid of instantaneous "
            + ("NVML samples. NVIDIA scopes the counter to Volta and newer, so a Pascal "
               "part such as a P100 lands here."
               if source == "nvml_integrated_samples" else
               "nvidia-smi samples. NVML is not installed, so whether this part has an "
               "energy counter is unknown; pip install nvidia-ml-py to find out."))
    else:
        return None
    return out


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
