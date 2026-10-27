"""Was anything else using this machine while we measured?

``membw.STABILITY_THRESHOLD`` catches competition that comes and goes, because that makes
the repetitions disagree. The ``membw`` docstring already names the case it cannot catch,
and gives this lab's own instance of it:

    Best-of-N cannot see *constant* background load: the same machine that measures
    50.7 GB/s idle measured 14.1 GB/s while a package manager ran.

A second instance turned up on 27 October 2026, which is what prompted writing the detector
rather than only the warning. On a Galaxy M21 (Exynos 9611, Android 12) the same read
kernels with the same arguments in the same power state measured 6.35 GB/s while Android
was running its background application optimization job and 9.32 GB/s once the device was
quiet. The contended reading is 32% low. Stability was 0.85 contended and 0.87 quiet, both
comfortably above the 0.75 flag, because the job ran for the whole measurement and so
depressed every repetition equally. Nothing in either output marked the low number as wrong.

Two details make that worth guarding against rather than filing under bad luck. The job ran
at nice 10, so it looks harmless to a scheduler-shaped intuition, but being nice yields the
CPU and not memory bandwidth or cache, which is all a bandwidth measurement cares about.
And Android schedules it when the device is idle and charging, which is precisely the state
a person creates in order to benchmark a phone.

So this measures the quantity directly instead of inferring it from scatter: of all the CPU
time the machine spent during our measurement window, how much went to somebody else.

Method, and why it needs no notion of clock rate or core count. :func:`cpu.snapshot` already
returns the aggregate ``(idle, total)`` counter pair, where ``total`` is summed over every
logical processor. So over a window, ``total`` advanced by exactly the machine's whole CPU
capacity, in whatever units the platform counts in: jiffies from ``/proc/stat`` on Linux and
Android, 100 ns ticks from ``GetSystemTimes`` on Windows. Dividing by it cancels the units
out, which is why this works on all three without a per-platform constant and without
needing ``SC_CLK_TCK``, ``os.cpu_count()`` or the elapsed wall time.

What a flag does and does not mean. It says the machine was busy, not that the ceiling is
wrong, and those are different claims. Two CPU-only spinners on an otherwise idle Raspberry
Pi 5 take the foreign fraction to 0.50 and leave the measured ceiling unchanged at
13.6 GB/s, because that board reaches its ceiling with one or two threads and the spinners
never touch memory. The package manager and the Android compiler both cost dearly because
both were streaming. Treat a flag as a reason to re-measure and compare, never as a
correction factor.

Observed values, so the 0.10 threshold can be calibrated against something:

* idle Raspberry Pi 5, nothing else running                       0.017
* Raspberry Pi 5 with two CPU-only spinners                       0.497
* Windows laptop during ordinary desktop use                      0.205 to 0.228
* Galaxy M21 running Android's background dexopt job              around 0.6

The Windows figure is worth dwelling on, because it is the common case rather than the
exotic one. On that machine two back-to-back ceiling measurements came out at 52.8 and
57.8 GB/s, 9% apart, while ``stability`` reported 0.976 and 0.977 for them. Neither run
looked unstable, both were wrong by some unknown amount, and the foreign fraction is the
only output that said so.

One practical consequence of the non-monotonic Windows counters: over a window of tens of
milliseconds, roughly a third of readings come back ``None`` because a counter moved
backwards within the window. Over a real ceiling measurement, which runs for seconds, that
did not happen in testing. So this is a reliable check at the scale it is used at and a
noisy one if called around something very short.
"""

from __future__ import annotations

import platform
from typing import Any, Dict, List, Optional, Tuple

from . import cpu

# Fraction of the machine's CPU capacity consumed by other processes during the
# measurement, above which the ceiling should not be trusted. An otherwise idle board sits
# in the low percents. The Android case sat near 0.6.
CONTENTION_THRESHOLD = 0.10


def _linux_own_cpu_time() -> Optional[int]:
    """This process's CPU time in the same jiffies ``/proc/stat`` counts in.

    utime and stime are fields 14 and 15 of ``/proc/self/stat`` and they aggregate the
    whole thread group, so the measurement's own worker threads land here rather than being
    mistaken for somebody else's work. The fields are located relative to the final ')'
    because an executable name can contain spaces and brackets, which defeats splitting the
    whole line.
    """
    try:
        with open("/proc/self/stat", "r", errors="replace") as fh:
            line = fh.read()
    except OSError:
        return None
    close = line.rfind(")")
    if close < 0:
        return None
    rest = line[close + 2:].split()
    # rest[0] is field 3 (state), so field 14 is rest[11] and field 15 is rest[12]
    if len(rest) < 13:
        return None
    try:
        return int(rest[11]) + int(rest[12])
    except ValueError:
        return None


def _windows_own_cpu_time() -> Optional[int]:
    """This process's kernel + user time in the same 100 ns ticks ``GetSystemTimes`` uses.

    ``GetProcessTimes`` covers every thread of the process, including ones that have since
    exited, which is what is wanted: the thread pool's time must not count as foreign.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class FILETIME(ctypes.Structure):
            _fields_ = [("dwLowDateTime", wintypes.DWORD),
                        ("dwHighDateTime", wintypes.DWORD)]

        # The signatures have to be declared. GetCurrentProcess returns the pseudo-handle
        # (HANDLE)-1, and without an explicit restype ctypes hands it back as a 32-bit
        # c_int, which does not widen to a 64-bit HANDLE on x64. GetProcessTimes then fails
        # and sets no last-error, so it looks like an unsupported platform rather than a
        # calling-convention mistake.
        kernel32 = ctypes.WinDLL("kernel32")
        LPFILETIME = ctypes.POINTER(FILETIME)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE, LPFILETIME, LPFILETIME,
                                             LPFILETIME, LPFILETIME]

        creation, exit_, kernel, user = FILETIME(), FILETIME(), FILETIME(), FILETIME()
        ok = kernel32.GetProcessTimes(
            kernel32.GetCurrentProcess(), ctypes.byref(creation), ctypes.byref(exit_),
            ctypes.byref(kernel), ctypes.byref(user)
        )
        if not ok:
            return None

        def as_int(ft: "FILETIME") -> int:
            return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

        return as_int(kernel) + as_int(user)
    except Exception:
        return None


def own_cpu_time() -> Optional[int]:
    """CPU time used by this process, in the platform's own counter units."""
    if platform.system() == "Windows":
        return _windows_own_cpu_time()
    return _linux_own_cpu_time()


def snapshot() -> Optional[Dict[str, Any]]:
    """Capture what is needed to work out foreign CPU use later. None when unsupported."""
    machine = cpu.snapshot()
    own = own_cpu_time()
    if machine is None or own is None:
        return None
    idle, total = machine[0]
    return {"idle": idle, "total": total, "own": own}


def foreign_fraction(before: Optional[Dict[str, Any]],
                     after: Optional[Dict[str, Any]]) -> Optional[float]:
    """Share of the machine's CPU capacity used by other processes between two snapshots.

    1.0 would mean every core was fully busy with somebody else's work for the whole
    window. Returns None when a snapshot is missing or the counters did not advance.
    """
    # `is None` rather than falsiness: None is the only value that means "could not check",
    # and a check that cannot run must not look like a check that passed.
    if before is None or after is None:
        return None
    capacity = after["total"] - before["total"]
    idle_delta = after["idle"] - before["idle"]
    own_delta = after["own"] - before["own"]
    # These counters are not strictly monotonic on Windows. GetSystemTimes sums per-processor
    # times, and cores being parked and unparked makes the sum go backwards now and then: on
    # one laptop `total` fell by 0.58 s worth of ticks between two reads a fraction of a
    # second apart. A regression inside a single interface means the window cannot be
    # measured, so the honest answer is None. Returning 0.0 here would be far worse than
    # returning nothing, because 0.0 reads as "checked, and the machine was idle".
    if capacity <= 0 or idle_delta < 0 or own_delta < 0:
        return None
    busy = capacity - idle_delta
    if busy < 0:
        return None
    foreign = busy - own_delta
    # Unlike the above, this subtraction crosses two interfaces read a moment apart, so a
    # small negative is ordinary rounding rather than evidence of anything.
    if foreign < 0:
        foreign = 0
    return min(1.0, foreign / capacity)


def process_table() -> Optional[Dict[int, Tuple[str, int]]]:
    """pid -> (name, cpu time), for naming whatever is competing.

    Linux and Android only. Windows returns None: the fraction above is still reported
    there, just without names, which is the useful half.
    """
    if platform.system() == "Windows":
        return None
    import os
    try:
        pids = [int(e) for e in os.listdir("/proc") if e.isdigit()]
    except (OSError, ValueError):
        return None
    out: Dict[int, Tuple[str, int]] = {}
    own = os.getpid()
    for pid in pids:
        if pid == own:
            continue
        try:
            with open(f"/proc/{pid}/stat", "r", errors="replace") as fh:
                line = fh.read()
        except OSError:
            continue  # the process exited between listdir and open, which is routine
        open_paren = line.find("(")
        close = line.rfind(")")
        if open_paren < 0 or close < open_paren:
            continue
        rest = line[close + 2:].split()
        if len(rest) < 13:
            continue
        try:
            out[pid] = (line[open_paren + 1:close], int(rest[11]) + int(rest[12]))
        except ValueError:
            continue
    return out


def top_foreign(before: Optional[Dict[int, Tuple[str, int]]],
                after: Optional[Dict[int, Tuple[str, int]]],
                limit: int = 3) -> List[Tuple[str, float]]:
    """The processes that used the most CPU between two process tables, as (name, time).

    A process that started during the window counts all of its usage, which is the honest
    reading: it was not there before and it was busy after. A reused pid is treated as new
    rather than credited with the previous owner's time.
    """
    if before is None or after is None:
        return []
    deltas: List[Tuple[str, float]] = []
    for pid, (name, used) in after.items():
        prior = before.get(pid)
        delta = used - prior[1] if prior and prior[0] == name else used
        if delta > 0:
            deltas.append((name, float(delta)))
    deltas.sort(key=lambda kv: kv[1], reverse=True)
    return deltas[:limit]


def describe(fraction: Optional[float],
             offenders: Optional[List[Tuple[str, float]]] = None) -> Optional[str]:
    """A warning for a measurement taken on a busy machine, or None when it looks clean."""
    if fraction is None or fraction < CONTENTION_THRESHOLD:
        return None
    names = ", ".join(name for name, _ in (offenders or [])[:3])
    who = f" The busiest were: {names}." if names else ""
    return (
        f"other processes used {100 * fraction:.0f}% of this machine's CPU capacity while "
        f"the bandwidth ceiling was being measured, so this ceiling, and every utilisation "
        f"percentage derived from it, should be treated as unreliable.{who} What it costs "
        f"depends on what the other work was doing: a package manager cost this lab a "
        f"factor of 3.6 once, and an Android compiler job cost 32%, while processes that "
        f"only burn CPU on otherwise spare cores can cost nothing at all. Steady "
        f"competition does not show up as instability, because it depresses every "
        f"repetition equally, so a healthy stability figure does not rule this out. Re-run "
        f"on an idle machine and compare."
    )


def describe_capability() -> Dict[str, bool]:
    """Whether this machine can answer the question at all. Feeds the capability model."""
    return {"cpu_contention": snapshot() is not None}
