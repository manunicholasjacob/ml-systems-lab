"""Per-core CPU utilisation on Windows (issue #2).

Windows has no /proc/stat. The aggregate comes from GetSystemTimes and the per-core
figures from NtQuerySystemInformation(SystemProcessorPerformanceInformation), and the
per-core list is only reported when, pooled, it gives the aggregate's busy fraction.

The first group of tests runs anywhere, off a snapshot pair recorded on a 20-thread
Windows laptop with one busy loop pinned to logical processor 0. The second group needs
Windows and reads the live counters; the psutil cross-check also needs psutil.
"""

import json
import multiprocessing
import os
import platform
import time
from pathlib import Path

import pytest

from mlsyslab.telemetry import cpu

RECORDED = Path(__file__).parent / "recorded" / "windows_cpu_times_20_logical_2026-10-13.json"
on_windows = pytest.mark.skipif(platform.system() != "Windows", reason="needs Windows")


def _recorded():
    rec = json.loads(RECORDED.read_text(encoding="utf-8"))

    def snap(side):
        return (tuple(rec[side]["aggregate"]), [tuple(c) for c in rec[side]["per_core"]])

    return rec, snap("before"), snap("after")


def _psutil_style_busy(before, after):
    """psutil's per-core busy percent from recorded cpu_times, the way psutil computes it.

    psutil counts interrupt and DPC time as separate fields on top of user and system, and
    its cpu_percent() sums every field into the total. Kernel time already contains them,
    so psutil reads about a point higher than the raw counters.
    """
    out = []
    for x, y in zip(before, after):
        d = {k: y[k] - x[k] for k in x}
        total = sum(d.values())
        out.append(100.0 * (total - d["idle"]) / total if total > 0 else None)
    return out


# ------------------------------------------------------------- recorded, runs anywhere

def test_recorded_pair_reports_every_logical_processor():
    rec, before, after = _recorded()
    util = cpu.utilisation(before, after)
    assert util["per_core"] is not None
    assert len(util["per_core"]) == 20


def test_recorded_pair_shows_the_pinned_core_saturated():
    _, before, after = _recorded()
    util = cpu.utilisation(before, after)
    assert util["per_core"][0] >= 99.0
    # And it is the busiest one: the core the load was pinned to, not some other.
    assert util["per_core"][0] == max(util["per_core"])


def test_recorded_per_core_pools_to_the_getsystemtimes_aggregate():
    _, before, after = _recorded()
    idle = sum(a[0] - b[0] for b, a in zip(before[1], after[1]))
    total = sum(a[1] - b[1] for b, a in zip(before[1], after[1]))
    pooled = 100.0 * (1 - idle / total)
    mean = cpu.utilisation(before, after)["mean"]
    assert abs(pooled - mean) < 0.5


def test_recorded_per_core_matches_psutil_over_the_same_window():
    rec, before, after = _recorded()
    ours = cpu.utilisation(before, after)["per_core"]
    theirs = _psutil_style_busy(rec["psutil_cpu_times_before"], rec["psutil_cpu_times_after"])
    diffs = [abs(o - t) for o, t in zip(ours, theirs)]
    # The two reads bracket each other by microseconds, and psutil adds interrupt and DPC
    # time on top, so a per-core gap of a few points is expected, not a systematic offset.
    assert max(diffs) < 6.0
    assert sum(diffs) / len(diffs) < 2.5


def test_per_core_that_does_not_add_up_is_dropped_not_reported():
    _, before, after = _recorded()
    # Corrupt one core so the per-core counters no longer pool to the aggregate.
    broken_after = (after[0], list(after[1]))
    idle, total = broken_after[1][3]
    broken_after[1][3] = (idle + 50_000_000, total + 50_000_000)
    util = cpu.utilisation(before, broken_after)
    assert util["mean"] is not None
    assert util["per_core"] is None


def test_synthetic_pair_that_agrees_keeps_its_per_core_list():
    before = ((100, 1000), [(50, 500), (50, 500)])
    after = ((130, 1100), [(80, 550), (50, 550)])
    assert cpu.utilisation(before, after)["per_core"] == [40.0, 100.0]


# ------------------------------------------------------------- live, Windows only

@on_windows
def test_live_snapshot_has_one_entry_per_logical_processor():
    snap = cpu.snapshot()
    assert snap is not None
    assert len(snap[1]) == os.cpu_count()


@on_windows
def test_live_per_core_agrees_with_the_aggregate():
    # The counters tick every 15.6 ms and the two interfaces are read a moment apart, so
    # a window of a few ticks can legitimately disagree by more than the tolerance. A
    # second is long enough that they cannot.
    before = cpu.snapshot()
    time.sleep(1.0)
    after = cpu.snapshot()
    util = cpu.utilisation(before, after)
    assert util["mean"] is not None
    assert util["per_core"] is not None, "per-core disagreed with GetSystemTimes"
    assert len(util["per_core"]) == os.cpu_count()


def _spin(seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


@on_windows
def test_live_per_core_matches_psutil_and_sees_a_pinned_load():
    psutil = pytest.importorskip("psutil")
    proc = multiprocessing.Process(target=_spin, args=(3.0,))
    proc.start()
    try:
        psutil.Process(proc.pid).cpu_affinity([0])
        time.sleep(0.3)
        pb = [x._asdict() for x in psutil.cpu_times(percpu=True)]
        before = cpu.snapshot()
        time.sleep(1.0)
        after = cpu.snapshot()
        pa = [x._asdict() for x in psutil.cpu_times(percpu=True)]
    finally:
        proc.join()
    ours = cpu.utilisation(before, after)["per_core"]
    theirs = _psutil_style_busy(pb, pa)
    assert ours is not None and len(ours) == len(theirs)
    assert ours[0] >= 90.0
    diffs = [abs(o - t) for o, t in zip(ours, theirs) if t is not None]
    assert sum(diffs) / len(diffs) < 5.0
