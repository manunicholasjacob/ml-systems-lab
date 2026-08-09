"""CPU utilisation, sampled cheaply enough not to distort what it measures.

Utilisation is a difference between two counter reads, not an instantaneous value, so the
API is a pair of snapshots and a function that turns them into percentages. Linux reads
``/proc/stat``; Windows calls ``GetSystemTimes`` through ctypes, which is the same idea
without the file.
"""

from __future__ import annotations

import platform
from typing import Dict, List, Optional, Tuple

# (idle_jiffies, total_jiffies) for the aggregate, plus the same per core.
Snapshot = Tuple[Tuple[int, int], List[Tuple[int, int]]]


def _linux_snapshot() -> Optional[Snapshot]:
    try:
        with open("/proc/stat", "r") as fh:
            lines = fh.readlines()
    except OSError:
        return None

    aggregate: Optional[Tuple[int, int]] = None
    per_core: List[Tuple[int, int]] = []
    for line in lines:
        if not line.startswith("cpu"):
            continue
        parts = line.split()
        label = parts[0]
        try:
            values = [int(v) for v in parts[1:]]
        except ValueError:
            continue
        if len(values) < 4:
            continue
        # user nice system idle iowait irq softirq steal ...
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        if label == "cpu":
            aggregate = (idle, total)
        else:
            per_core.append((idle, total))
    return (aggregate, per_core) if aggregate else None


def _windows_snapshot() -> Optional[Snapshot]:
    try:
        import ctypes
        from ctypes import wintypes

        class FILETIME(ctypes.Structure):
            _fields_ = [("dwLowDateTime", wintypes.DWORD),
                        ("dwHighDateTime", wintypes.DWORD)]

        def as_int(ft: "FILETIME") -> int:
            return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

        idle_ft, kernel_ft, user_ft = FILETIME(), FILETIME(), FILETIME()
        ok = ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle_ft), ctypes.byref(kernel_ft), ctypes.byref(user_ft)
        )
        if not ok:
            return None
        idle = as_int(idle_ft)
        # Kernel time already includes idle time on Windows, so total is kernel + user.
        total = as_int(kernel_ft) + as_int(user_ft)
        return ((idle, total), [])
    except Exception:
        return None


def snapshot() -> Optional[Snapshot]:
    if platform.system() == "Windows":
        return _windows_snapshot()
    return _linux_snapshot()


def _percent(before: Tuple[int, int], after: Tuple[int, int]) -> Optional[float]:
    idle_delta = after[0] - before[0]
    total_delta = after[1] - before[1]
    if total_delta <= 0:
        return None
    busy = 100.0 * (1.0 - idle_delta / total_delta)
    # Counters can jitter by a hair across a read; clamp rather than emit -0.3%.
    return max(0.0, min(100.0, busy))


def utilisation(before: Snapshot, after: Snapshot) -> Dict[str, object]:
    """Percent busy between two snapshots, aggregate and per core where available."""
    result: Dict[str, object] = {"mean": None, "per_core": None}
    if not before or not after:
        return result
    result["mean"] = _percent(before[0], after[0])
    if before[1] and after[1] and len(before[1]) == len(after[1]):
        cores = [_percent(b, a) for b, a in zip(before[1], after[1])]
        if any(c is not None for c in cores):
            result["per_core"] = [round(c, 1) if c is not None else None for c in cores]
    return result


def process_peak_rss_bytes() -> Optional[int]:
    """Peak resident set size of this process and its waited-for children.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS, a difference that silently
    produces a 1024x error if you do not handle it.
    """
    try:
        import resource
    except ImportError:
        return None
    try:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        own = resource.getrusage(resource.RUSAGE_SELF)
    except (OSError, ValueError):
        return None
    peak = max(usage.ru_maxrss, own.ru_maxrss)
    if peak <= 0:
        return None
    return peak if platform.system() == "Darwin" else peak * 1024


def describe() -> dict:
    return {"cpu_utilisation": snapshot() is not None,
            "peak_rss": process_peak_rss_bytes() is not None}
