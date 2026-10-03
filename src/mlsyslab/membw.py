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

The same discipline is applied to accelerator memory by :func:, which
reaches the device through torch and times on the stream with CUDA events rather than
around an asynchronous launch. It reports the ceiling as absent when there is no device,
because a GPU roofline built on a datasheet number has the same failure mode as a CPU one.

Requires numpy on the machine being measured, and torch for the device path. Run directly
for JSON:

    python -m mlsyslab.membw --json
    python -m mlsyslab.membw --device --json
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


def positive_int(text: str) -> int:
    """argparse type: a whole number of at least 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}")
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _require_reps(reps: int) -> None:
    # With no repetitions every kernel scores 0.0, and the result is a 0.00 GB/s ceiling
    # with no kernel and no thread count that still reads like a measurement, and that
    # `mlsys membw` then offers as the device's dram_peak_GBs.
    if reps < 1:
        raise ValueError(f"reps must be at least 1, got {reps}")


def measure(working_set_mb: Optional[int] = None,
            thread_counts: Optional[List[int]] = None,
            reps: int = 5) -> Dict[str, Any]:
    """Measure achievable memory bandwidth. ``peak_read_GBs`` is the headline."""
    _require_reps(reps)
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


# --------------------------------------------------------------- accelerator memory

# A GPU roofline needs a measured HBM or GDDR read ceiling for exactly the reason the CPU
# one does: `dram_peak_GBs` is the denominator of every utilisation percentage, and a
# datasheet figure turns a wrong assumption into a confident-looking number.
#
# The discipline is the same as the CPU path above and is not repeated by accident:
# read-only kernels only, copy measured for context and excluded because write-allocate
# inflates it, best of N across configurations, every repetition kept, and the
# median-to-best stability flag. On a rented GPU the stability flag matters more than it
# does on a laptop, because another tenant may be on the same host and best-of-N cannot
# see steady contention.
#
# torch is the vehicle rather than a dependency. It is preinstalled on Kaggle and Colab,
# which is where this path is meant to run, and it is imported lazily so a device without
# it reports the ceiling as absent instead of failing.

DEVICE_WORKING_SET_MB = 1024
DEVICE_MIN_FREE_FRACTION = 0.25


class AcceleratorUnavailable(RuntimeError):
    """No accelerator this module knows how to measure."""


def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def describe_no_cuda(torch) -> str:
    """Say which of the two reasons torch cannot see a GPU, because they differ.

    Either the session has no accelerator attached, or torch itself is a CPU-only build.
    The second is the one that wastes an afternoon: pip will happily replace a CUDA build
    with a CPU wheel while installing something else, and nothing says so.
    """
    built_for_cuda = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        count = torch.cuda.device_count()
    except Exception:
        count = 0
    if not built_for_cuda:
        return (f"torch {getattr(torch, '__version__', '?')} is a CPU-only build, so it "
                f"cannot see a GPU whatever is attached. torch.version.cuda is None. "
                f"Reinstall a CUDA build, and check afterwards that it survived: on a "
                f"hosted notebook, installing another package can pull a CPU wheel over "
                f"the top of it.")
    return (f"torch {getattr(torch, '__version__', '?')} is built for CUDA "
            f"{built_for_cuda} and reports {count} devices, so the runtime has no "
            f"accelerator attached. On Kaggle or Colab, pick a GPU accelerator for the "
            f"session and run this again. nvidia-smi in a shell cell is the quickest "
            f"check.")


def device_available() -> bool:
    torch = _torch()
    return bool(torch and torch.cuda.is_available())


def _device_working_set_bytes(torch, requested_mb: Optional[int]) -> int:
    """Big enough to defeat the last-level cache, small enough to leave the card usable."""
    free, total = torch.cuda.mem_get_info()
    ceiling = int(free * (1.0 - DEVICE_MIN_FREE_FRACTION))
    wanted = (requested_mb or DEVICE_WORKING_SET_MB) * 1024 * 1024
    chosen = min(wanted, ceiling)
    if chosen < MIN_WORKING_SET_MB * 1024 * 1024:
        raise AcceleratorUnavailable(
            f"only {free / 1e6:.0f} MB free on the device, which is not enough to measure "
            f"a ceiling without evicting whatever else is resident")
    return chosen


def measure_device(working_set_mb: Optional[int] = None, reps: int = 7,
                   index: int = 0) -> Dict[str, Any]:
    """Achievable device-memory read bandwidth, measured on the accelerator itself.

    Raises :class:`AcceleratorUnavailable` rather than returning a guess when there is no
    device or no torch.
    """
    _require_reps(reps)
    torch = _torch()
    if torch is None:
        raise AcceleratorUnavailable(
            "torch is not installed, and it is what this path uses to reach the device. "
            "Install it, or declare dram_peak_GBs in the device config and say where the "
            "number came from.")
    if not torch.cuda.is_available():
        raise AcceleratorUnavailable(describe_no_cuda(torch))

    torch.cuda.set_device(index)
    nbytes = _device_working_set_bytes(torch, working_set_mb)
    n = nbytes // 4  # float32

    a = torch.ones(n, dtype=torch.float32, device="cuda")
    b = torch.ones(n, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(a)

    def timed(fn, moved: int) -> List[float]:
        """Rates in GB/s, one per repetition, timed on the device rather than the host.

        CUDA events time the stream itself. A host-side timer around an asynchronous
        launch measures the launch, which on a small kernel is most of what it reports.
        """
        out: List[float] = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        fn()  # warm the kernel and the allocator out of the timed region
        torch.cuda.synchronize()
        for _ in range(reps):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)
            if ms > 0:
                out.append(moved / (ms / 1000.0) / 1e9)
        return out

    try:
        raw = {
            "sum": timed(lambda: a.sum(), a.numel() * 4),
            "max": timed(lambda: a.max(), a.numel() * 4),
            "dot": timed(lambda: torch.dot(a, b), 2 * a.numel() * 4),
            "copy": timed(lambda: dst.copy_(a), 2 * a.numel() * 4),
        }
    finally:
        del a, b, dst
        torch.cuda.empty_cache()

    best = {k: max(v, default=0.0) for k, v in raw.items()}
    read_kernels = ("sum", "max", "dot")
    peak_kernel = max(read_kernels, key=lambda k: best[k])
    peak = best[peak_kernel]
    winning = raw[peak_kernel]
    stability = (_median(winning) / peak) if peak and winning else None

    props = torch.cuda.get_device_properties(index)
    out: Dict[str, Any] = {
        "status": "ok",
        "peak_read_GBs": round(peak, 3),
        "best_kernel": peak_kernel,
        "copy_GBs": round(best["copy"], 3),
        "working_set_mb": round(nbytes / 1024 / 1024),
        "reps": reps,
        "stability": round(stability, 4) if stability is not None else None,
        "by_kernel_GBs": {k: [round(x, 3) for x in v] for k, v in raw.items()},
        "device_index": index,
        "device_name": props.name,
        "device_total_MiB": round(props.total_memory / 1048576),
        "compute_capability": f"{props.major}.{props.minor}",
        "torch_version": torch.__version__,
        "method": ("torch read-only reductions on the device, timed with CUDA events; "
                   "ceiling is the best read-only kernel and is a measured lower bound"),
        "source": "measured",
        "target": "device",
    }
    if stability is not None and stability < STABILITY_THRESHOLD:
        out["unstable"] = True
        out["warning"] = (
            f"device bandwidth was unstable: repetitions varied by "
            f"{100 * (1 - stability):.0f}% around the best result. On a shared host that "
            f"usually means another tenant. Re-run, and do not put this number in a "
            f"config until it repeats.")
    return out


def device_context() -> Dict[str, Any]:
    """What the machine says about its accelerator, for a record that has to travel."""
    torch = _torch()
    ctx: Dict[str, Any] = {"torch_installed": torch is not None}
    if torch is None:
        return ctx
    ctx["torch_version"] = getattr(torch, "__version__", None)
    ctx["torch_cuda_build"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        ctx["cuda_available"] = bool(torch.cuda.is_available())
        ctx["device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        ctx["cuda_available"] = False
        ctx["probe_error"] = f"{type(exc).__name__}: {exc}"
    return ctx


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="mlsyslab.membw")
    parser.add_argument("--mb", type=positive_int, help="working set in MB")
    parser.add_argument("--reps", type=positive_int, default=None,
                        help="repetitions per kernel (default: 5 on the host, 7 on "
                             "an accelerator)")
    parser.add_argument("--json", action="store_true", help="bare JSON, no sentinels")
    parser.add_argument("--device", action="store_true",
                        help="measure the accelerator's memory instead of the host's")
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        if args.device:
            result = measure_device(working_set_mb=args.mb, reps=args.reps or 7,
                                    index=args.device_index)
        else:
            result = measure(working_set_mb=args.mb, reps=args.reps or 5)
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        if args.device:
            # A one-shot notebook cell has to diagnose itself. Without this the record is
            # an error string with no way to tell an unattached GPU from a CPU-only wheel.
            result["context"] = device_context()

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        sys.stdout.write(RESULT_BEGIN + "\n")
        sys.stdout.write(json.dumps(result))
        sys.stdout.write("\n" + RESULT_END + "\n")
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
