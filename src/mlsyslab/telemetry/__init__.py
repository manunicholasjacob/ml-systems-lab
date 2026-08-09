"""One sampling thread for every physical signal, because the sampler is not free.

The temptation is a thread per metric. That is wrong here. On a Raspberry Pi, reading the
PMIC spawns a process, and running that at 50 Hz slowed decode by roughly 6x: the
measurement destroyed the thing being measured. So there is exactly one thread, it ticks
at a deliberately low rate, and the expensive source is opt-in.

Two consequences worth stating plainly, because they are methodology rather than code:

* **Throughput and power should not be read from the same run.** Measure throughput with
  the sampler off, then power in a second run with it on. The two runs are comparable
  because the workload is deterministic; a single sampled run is not comparable to
  anything.
* **Restrict the summary to the active window.** Model load can dominate a short run, and
  averaging power across it reports the loader rather than the inference.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from . import cpu, dvfs, power, thermal

DEFAULT_SAMPLE_HZ = 10.0


def capabilities() -> Dict[str, bool]:
    """What this machine can actually measure. Drives the capability model on Device."""
    caps: Dict[str, bool] = {}
    caps.update(thermal.describe())
    caps.update(power.describe())
    caps.update(cpu.describe())
    caps.update(dvfs.describe())
    return caps


class TelemetrySession:
    """Samples temperature, clock and optionally power on one low-rate thread."""

    def __init__(self, sample_hz: float = DEFAULT_SAMPLE_HZ, sample_power: bool = True):
        self.sample_hz = max(0.5, float(sample_hz))
        self.period = 1.0 / self.sample_hz
        # Power is the only expensive source, so it is the only one that can be declined.
        self.sample_power = bool(sample_power) and power.pmic_available()
        self.sample_rapl = bool(sample_power) and power.rapl_available()

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: List[Dict[str, Any]] = []
        self._cpu_before = None
        self._cpu_after = None
        self._rapl_before: Optional[int] = None
        self._rapl_after: Optional[int] = None
        self._throttled: Optional[bool] = None
        self._throttle_flags: Optional[str] = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> "TelemetrySession":
        self._samples = []
        self._stop.clear()
        self._cpu_before = cpu.snapshot()
        self._rapl_before = power.read_rapl_uj() if self.sample_rapl else None
        self._throttled, self._throttle_flags = thermal.read_throttle_flags()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> "TelemetrySession":
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._cpu_after = cpu.snapshot()
        self._rapl_after = power.read_rapl_uj() if self.sample_rapl else None
        # Re-read throttle state: a run that throttled midway is the case we care about.
        throttled_now, flags_now = thermal.read_throttle_flags()
        if throttled_now:
            self._throttled, self._throttle_flags = throttled_now, flags_now
        return self

    def __enter__(self) -> "TelemetrySession":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -------------------------------------------------------------------- sampling

    def _loop(self) -> None:
        while not self._stop.is_set():
            tick: Dict[str, Any] = {"t": time.perf_counter()}
            temp = thermal.read_temp_c()
            if temp is not None:
                tick["temp_c"] = temp
            freqs = thermal.cpu_freq_khz()
            if freqs:
                tick["freq_khz"] = sum(freqs) / len(freqs)
            if self.sample_power:
                reading = power.read_pmic()
                if reading:
                    tick["w_total"], tick["w_core"], tick["w_dram"] = reading
            self._samples.append(tick)
            # Sleep the remainder of the period rather than a fixed period, so an
            # expensive PMIC read does not silently halve the effective rate.
            elapsed = time.perf_counter() - tick["t"]
            time.sleep(max(0.0, self.period - elapsed))

    # --------------------------------------------------------------------- results

    def _window(self, t0: Optional[float], t1: Optional[float]) -> List[Dict[str, Any]]:
        lo = t0 if t0 is not None else float("-inf")
        hi = t1 if t1 is not None else float("inf")
        return [s for s in self._samples if lo <= s["t"] <= hi]

    def summary(
        self,
        t0: Optional[float] = None,
        t1: Optional[float] = None,
        tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Telemetry-shaped dict for the window [t0, t1], suitable for a RunRecord.

        Pass the active window to exclude model load. Pass ``tokens`` to also get energy
        per token, which is the number that actually compares across model sizes.
        """
        window = self._window(t0, t1)
        out: Dict[str, Any] = {
            "sample_hz": self.sample_hz,
            "sample_count": len(window),
            "throttled": self._throttled,
            "throttle_flags": self._throttle_flags,
        }

        temps = [s["temp_c"] for s in window if "temp_c" in s]
        if temps:
            out["temp_c_start"] = temps[0]
            out["temp_c_mean"] = sum(temps) / len(temps)
            out["temp_c_max"] = max(temps)

        freqs = [s["freq_khz"] for s in window if "freq_khz" in s]
        if freqs:
            out["freq_khz_mean"] = sum(freqs) / len(freqs)
            out["freq_khz_min"] = int(min(freqs))
            out["freq_khz_max"] = int(max(freqs))

        watts = [(s["t"], s["w_total"]) for s in window if "w_total" in s]
        if watts:
            out["power_w_mean"] = sum(w for _, w in watts) / len(watts)
            energy = power.integrate_energy(watts)
            if energy is not None:
                out["energy_j"] = energy
                if tokens:
                    out["energy_per_token_mj"] = 1000.0 * energy / tokens
            core = [s["w_core"] for s in window if "w_core" in s]
            dram = [s["w_dram"] for s in window if "w_dram" in s]
            if core:
                out["power_w_core"] = sum(core) / len(core)
            if dram:
                out["power_w_dram"] = sum(dram) / len(dram)
        elif self._rapl_before is not None and self._rapl_after is not None:
            # RAPL covers the whole session rather than the window, so it is only used
            # when there is no per-sample source to slice.
            energy = power.rapl_delta_j(self._rapl_before, self._rapl_after)
            if energy > 0:
                out["energy_j"] = energy
                if tokens:
                    out["energy_per_token_mj"] = 1000.0 * energy / tokens

        util = cpu.utilisation(self._cpu_before, self._cpu_after)
        if util.get("mean") is not None:
            out["cpu_util_pct_mean"] = util["mean"]
        if util.get("per_core"):
            out["cpu_util_pct_per_core"] = util["per_core"]

        return {k: v for k, v in out.items() if v is not None}


__all__ = [
    "TelemetrySession", "capabilities", "cpu", "dvfs", "power", "thermal",
    "DEFAULT_SAMPLE_HZ",
]
