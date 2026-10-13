"""CPU utilisation, sampled cheaply enough not to distort what it measures.

Utilisation is a difference between two counter reads, not an instantaneous value, so the
API is a pair of snapshots and a function that turns them into percentages. Linux reads
``/proc/stat``; Windows calls ``GetSystemTimes`` for the aggregate and
``NtQuerySystemInformation(SystemProcessorPerformanceInformation)`` for each logical
processor, both through ctypes, which is the same idea without the file.

Per-core figures are only reported when, pooled, they give the same busy fraction as the
aggregate over the same interval (:func:`utilisation`). Absent is better than invented.
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


# NtQuerySystemInformation information class, from winternl.h.
SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION = 8
ALL_PROCESSOR_GROUPS = 0xFFFF
STATUS_SUCCESS = 0
# Largest gap, in percentage points, allowed between the aggregate and the per-core
# figures pooled over the same interval before the per-core list is dropped. The two are
# read a moment apart and both advance in 15.6 ms ticks, so they agree to under half a
# point over 0.5 s on a 20-thread laptop, but can differ by several points over 50 ms.
# A very short interval therefore loses its per-core list, which is the intended side of
# the trade: a run is seconds long, and absent is better than invented.
PER_CORE_AGREEMENT_PCT = 2.0


def _windows_per_core() -> List[Tuple[int, int]]:
    """(idle, kernel + user) per logical processor, in 100 ns units, or [] if unavailable.

    ``SystemProcessorPerformanceInformation`` only covers the calling thread's processor
    group, so on a machine with more than 64 logical processors it returns fewer entries
    than exist. That case reports [] rather than a partial list that would read as the
    whole machine.
    """
    import ctypes
    from ctypes import wintypes

    class SPPI(ctypes.Structure):
        _fields_ = [("IdleTime", ctypes.c_longlong),
                    ("KernelTime", ctypes.c_longlong),
                    ("UserTime", ctypes.c_longlong),
                    ("DpcTime", ctypes.c_longlong),
                    ("InterruptTime", ctypes.c_longlong),
                    ("InterruptCount", ctypes.c_ulong)]

    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.GetActiveProcessorCount.restype = wintypes.DWORD
    kernel32.GetActiveProcessorCount.argtypes = [wintypes.WORD]
    expected = int(kernel32.GetActiveProcessorCount(ALL_PROCESSOR_GROUPS))
    if expected <= 0:
        return []

    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtQuerySystemInformation.restype = ctypes.c_long
    ntdll.NtQuerySystemInformation.argtypes = [ctypes.c_ulong, ctypes.c_void_p,
                                               ctypes.c_ulong,
                                               ctypes.POINTER(ctypes.c_ulong)]
    buf = (SPPI * expected)()
    returned = ctypes.c_ulong(0)
    status = ntdll.NtQuerySystemInformation(SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION,
                                            ctypes.byref(buf), ctypes.sizeof(buf),
                                            ctypes.byref(returned))
    if status != STATUS_SUCCESS:
        return []
    if returned.value // ctypes.sizeof(SPPI) != expected:
        return []
    # As with GetSystemTimes, kernel time already includes idle time.
    return [(int(p.IdleTime), int(p.KernelTime) + int(p.UserTime)) for p in buf]


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
    except Exception:
        return None
    try:
        per_core = _windows_per_core()
    except Exception:
        per_core = []
    return ((idle, total), per_core)


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
        if not _per_core_agrees(before, after, result["mean"]):
            return result
        cores = [_percent(b, a) for b, a in zip(before[1], after[1])]
        if any(c is not None for c in cores):
            result["per_core"] = [round(c, 1) if c is not None else None for c in cores]
    return result


def _per_core_agrees(before: Snapshot, after: Snapshot, mean: Optional[float]) -> bool:
    """Do the per-core counters, pooled, give the aggregate's busy fraction?

    They count the same time through two different interfaces, so they must. If they do
    not, the per-core read is wrong in a way this code does not understand, and the list is
    dropped rather than reported.
    """
    if mean is None:
        return False
    pooled_before = (sum(c[0] for c in before[1]), sum(c[1] for c in before[1]))
    pooled_after = (sum(c[0] for c in after[1]), sum(c[1] for c in after[1]))
    pooled = _percent(pooled_before, pooled_after)
    return pooled is not None and abs(pooled - mean) <= PER_CORE_AGREEMENT_PCT


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
