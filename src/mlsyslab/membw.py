"""Memory-bandwidth ceiling measurement, runnable on any device in the lab.

``dram_peak_GBs`` drives every roofline-utilisation percentage in the analysis layer,
and a wrong ceiling silently turns into a wrong percentage. This module measures an
*achievable* streaming-read rate on the machine it runs on, so a device config can carry
a number the lab itself produced rather than one typed in from a datasheet.

Method (adapted from the validated llama-roofline implementation):

* three read-only kernels through numpy's C ufuncs with the GIL released, a thread pool
  over disjoint slices: ``sum`` (one stream), ``max`` (one stream, no accumulation
  dependency), ``dot`` (two streams through BLAS, usually the winner);
* ``copy`` is measured for context but excluded from the ceiling, because write-allocate
  inflates it relative to what decode actually does;
* the ceiling is the best read-only result across thread counts, and is a **lower bound**
  on the true peak, never an upper bound;
* every repetition is kept, and a median-to-best ratio below 0.75 flags the measurement
  as unstable. Best-of-N cannot see *constant* background load: the same machine that
  measures 50.7 GB/s idle measured 14.1 GB/s while a package manager ran.

Requires numpy on the machine being measured. Run directly for JSON:

    python -m mlsyslab.membw --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

RESULT_BEGIN = "===MLSYSLAB-MEMBW-BEGIN==="
RESULT_END = "===MLSYSLAB-MEMBW-END==="

# Big enough to blow past any last-level cache, small enough to be polite on a 2 GB Pi.
DEFAULT_WORKING_SET_MB = 512
MIN_WORKING_SET_MB = 64
STABILITY_THRESHOLD = 0.75


def choose_working_set_mb(requested: Optional[int] = None,
                          ram_bytes: Optional[int] = None) -> int:
    """A working set that exceeds cache but cannot push the machine into swap."""
    if requested:
        return max(MIN_WORKING_SET_MB, int(requested))
    mb = DEFAULT_WORKING_SET_MB
    if ram_bytes:
        # The copy kernel needs two buffers, so cap total at ~1/8 of RAM.
        mb = min(mb, max(MIN_WORKING_SET_MB, int(ram_bytes / 1e6 / 8)))
    return int(mb)


def _samples(fn, slices, nthreads: int, bytes_moved: int, reps: int) -> List[float]:
    out: List[float] = []
    with ThreadPoolExecutor(max_workers=nthreads) as pool:
        for _ in range(reps):
            t0 = time.perf_counter()
            list(pool.map(fn, slices))
            dt = time.perf_counter() - t0
            if dt > 0:
                out.append(bytes_moved / dt / 1e9)
    return out


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def measure(working_set_mb: Optional[int] = None,
            thread_counts: Optional[List[int]] = None,
            reps: int = 5) -> Dict[str, Any]:
    """Measure achievable memory bandwidth. ``peak_read_GBs`` is the headline."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "numpy is required to measure the bandwidth ceiling. Install it on this "
            "device, or declare dram_peak_GBs in the device config instead."
        ) from exc

    from .sysinfo import total_ram_bytes

    mb = choose_working_set_mb(working_set_mb, total_ram_bytes())
    nbytes = mb * 1024 * 1024
    logical = os.cpu_count() or 4
    if not thread_counts:
        thread_counts = sorted({1, max(1, logical // 4), max(1, logical // 2), logical})

    n64 = nbytes // 8
    a64 = np.ones(n64, dtype=np.float64)
    b64 = np.empty_like(a64)
    n32 = nbytes // 4
    a32 = np.ones(n32, dtype=np.float32)
    b32 = np.ones(n32, dtype=np.float32)

    results: Dict[str, Dict[str, float]] = {}
    raw: Dict[str, Dict[str, List[float]]] = {}
    try:
        for nt in thread_counts:
            sl64 = [(i * n64 // nt, (i + 1) * n64 // nt) for i in range(nt)]
            sl32 = [(i * n32 // nt, (i + 1) * n32 // nt) for i in range(nt)]
            cell = {
                "sum": _samples(lambda s: float(a64[s[0]:s[1]].sum()),
                                sl64, nt, a64.nbytes, reps),
                "max": _samples(lambda s: float(a64[s[0]:s[1]].max()),
                                sl64, nt, a64.nbytes, reps),
                "dot": _samples(lambda s: float(np.dot(a32[s[0]:s[1]], b32[s[0]:s[1]])),
                                sl32, nt, 2 * a32.nbytes, reps),
                "copy": _samples(lambda s: b64.__setitem__(slice(s[0], s[1]),
                                                           a64[s[0]:s[1]]),
                                 sl64, nt, 2 * a64.nbytes, reps),
            }
            raw[str(nt)] = cell
            results[str(nt)] = {k: max(v, default=0.0) for k, v in cell.items()}
    finally:
        del a64, b64, a32, b32

    peak, best_kernel, best_threads = 0.0, None, None
    for nt, row in results.items():
        for kernel in ("sum", "max", "dot"):
            if row[kernel] > peak:
                peak, best_kernel, best_threads = row[kernel], kernel, int(nt)
    copy_peak = max((row["copy"] for row in results.values()), default=0.0)

    winning = raw.get(str(best_threads), {}).get(best_kernel, []) if best_kernel else []
    stability = (_median(winning) / peak) if peak and winning else None

    out: Dict[str, Any] = {
        "status": "ok",
        "peak_read_GBs": peak,
        "best_kernel": best_kernel,
        "best_threads": best_threads,
        "copy_GBs": copy_peak,
        "working_set_mb": mb,
        "reps": reps,
        "stability": stability,
        "by_threads": results,
        "method": ("numpy threaded read kernels (sum/max/dot); ceiling is the best "
                   "read-only kernel, a measured lower bound on true peak"),
    }
    if stability is not None and stability < STABILITY_THRESHOLD:
        out["unstable"] = True
        out["warning"] = (
            f"repetitions varied by {100 * (1 - stability):.0f}% around the best result, "
            "which usually means something else was using the machine. The ceiling is "
            "probably too low, so every utilisation percentage derived from it would be "
            "too high. Re-run on an idle machine."
        )
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="mlsyslab.membw")
    parser.add_argument("--mb", type=int, help="working set in MB")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="bare JSON, no sentinels")
    args = parser.parse_args(argv)

    try:
        result = measure(working_set_mb=args.mb, reps=args.reps)
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        sys.stdout.write(RESULT_BEGIN + "\n")
        sys.stdout.write(json.dumps(result))
        sys.stdout.write("\n" + RESULT_END + "\n")
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
