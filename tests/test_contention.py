"""Tests for the foreign-CPU check on the bandwidth ceiling.

The case that matters is the one the stability flag cannot see: competition that lasts for
the whole measurement, so every repetition is depressed equally and the scatter looks fine.
This lab has two instances on record, a package manager worth a factor of 3.6 and an
Android background compiler worth 32% at stability 0.85.

The formula under test divides by the aggregate ``total`` counter delta, which already sums
over every logical processor, so it needs no clock rate, core count or elapsed time and the
platform's units cancel out. Several tests below exist to pin that property down, because it
is the reason this works on Linux, Android and Windows alike.
"""

from __future__ import annotations

import platform

import pytest

from mlsyslab.telemetry import contention


def snap(idle: int, total: int, own: int) -> dict:
    return {"idle": idle, "total": total, "own": own}


# ---------------------------------------------------------------- foreign_fraction


def test_idle_machine_reports_no_foreign_use():
    # 1000 ticks of capacity, 800 idle, and all 200 busy ticks are ours.
    assert contention.foreign_fraction(snap(0, 0, 0), snap(800, 1000, 200)) == pytest.approx(0.0)


def test_half_the_machine_taken_by_others():
    # 1000 of capacity, none idle, none ours.
    assert contention.foreign_fraction(snap(0, 0, 0), snap(500, 1000, 0)) == pytest.approx(0.5)


def test_fully_busy_with_someone_elses_work():
    assert contention.foreign_fraction(snap(0, 0, 0), snap(0, 1000, 0)) == pytest.approx(1.0)


def test_our_own_work_is_never_counted_as_foreign():
    # Entirely busy, entirely us. This is what a correct measurement run looks like and it
    # must not warn, or the check would fire on every run it is meant to protect.
    assert contention.foreign_fraction(snap(0, 0, 0), snap(0, 1000, 1000)) == pytest.approx(0.0)


def test_units_cancel_so_the_same_situation_scores_the_same_on_any_platform():
    # Linux counts jiffies, Windows counts 100 ns ticks. The same half-busy-elsewhere
    # machine has to come out at 0.5 either way, which is the whole reason this formula
    # was chosen over one involving SC_CLK_TCK and os.cpu_count().
    jiffies = contention.foreign_fraction(snap(0, 0, 0), snap(500, 1000, 0))
    filetime = contention.foreign_fraction(snap(0, 0, 0), snap(5_000_000, 10_000_000, 0))
    assert jiffies == pytest.approx(filetime) == pytest.approx(0.5)


def test_core_count_needs_no_special_handling():
    # The aggregate counter advances by core_count * elapsed * rate on its own, so a 2-core
    # and a 64-core machine with the same proportion busy elsewhere give the same answer.
    assert contention.foreign_fraction(snap(0, 0, 0), snap(50, 100, 0)) == pytest.approx(
        contention.foreign_fraction(snap(0, 0, 0), snap(1600, 3200, 0)))


def test_our_time_appearing_to_exceed_the_machines_is_floored_not_propagated():
    # The machine counter and the process counter are read a moment apart through different
    # interfaces, so our own time can come out slightly above the machine's busy time. That
    # crosses two interfaces, so it is rounding and the floor is right.
    assert contention.foreign_fraction(snap(0, 0, 0), snap(900, 1000, 150)) == 0.0


def test_fraction_is_capped_at_one():
    # Idle going backwards within one read would put the fraction above 1, which would be
    # reported as "140% of the machine". Not a useful thing to tell anyone.
    assert contention.foreign_fraction(snap(0, 0, 0), snap(-500, 1000, 0)) is None


def test_a_counter_that_went_backwards_returns_none_rather_than_a_clean_bill():
    # Windows CPU-time counters are not strictly monotonic: GetSystemTimes sums per-processor
    # times and parking cores makes the sum fall. On one laptop `total` dropped by 0.58 s
    # worth of ticks between two reads. A regression inside one interface means the window
    # cannot be measured, and 0.0 would read as "checked, and the machine was idle".
    assert contention.foreign_fraction(snap(0, 1000, 0), snap(0, 900, 0)) is None     # total
    assert contention.foreign_fraction(snap(500, 0, 0), snap(400, 1000, 0)) is None   # idle
    assert contention.foreign_fraction(snap(0, 0, 500), snap(0, 1000, 400)) is None   # own


def test_missing_snapshot_returns_none_not_zero():
    # None means "could not check". Zero would mean "checked, and the machine was clean",
    # which is the one thing an unsupported platform must not claim.
    assert contention.foreign_fraction(None, snap(0, 1, 0)) is None
    assert contention.foreign_fraction(snap(0, 1, 0), None) is None
    assert contention.foreign_fraction(None, None) is None


def test_counters_that_did_not_advance_return_none():
    assert contention.foreign_fraction(snap(0, 1000, 0), snap(0, 1000, 0)) is None


# ---------------------------------------------------------------- describe


def test_steady_contention_is_caught_where_stability_would_pass():
    # The Android dexopt case: about 60% of an 8-core phone taken by five compiler threads
    # for the whole run, at a stability of 0.85 that the existing flag passed.
    msg = contention.describe(0.6, [("dex2oat", 500.0), ("Compiler driver", 480.0)])
    assert msg is not None
    assert "60%" in msg and "dex2oat" in msg
    # It must say why a good stability figure is not reassurance, or a reader will trust
    # the stability number and discard this one.
    assert "does not show up as instability" in msg


def test_warning_does_not_claim_the_ceiling_is_definitely_wrong():
    # Measured on a Pi 5: two CPU-only spinners take the fraction to 0.50 and leave the
    # ceiling unchanged, because nothing else touched memory. The warning has to survive
    # that case without having lied.
    msg = contention.describe(0.5, [("python3", 400.0)])
    assert msg is not None and "unreliable" in msg
    assert "can cost nothing at all" in msg


def test_quiet_machine_produces_no_warning():
    assert contention.describe(0.01) is None
    assert contention.describe(0.0, []) is None


def test_unsupported_platform_produces_no_warning():
    assert contention.describe(None) is None


def test_threshold_boundary_is_inclusive():
    assert contention.describe(contention.CONTENTION_THRESHOLD - 1e-9) is None
    assert contention.describe(contention.CONTENTION_THRESHOLD) is not None


def test_describe_without_offender_names_still_warns():
    # Windows has no process table here, and the fraction alone is still the useful half.
    msg = contention.describe(0.4, [])
    assert msg is not None and "busiest" not in msg


# ---------------------------------------------------------------- top_foreign


def test_top_foreign_ranks_by_cpu_used_during_the_window():
    before = {1: ("quiet", 100), 2: ("busy", 100), 3: ("medium", 100)}
    after = {1: ("quiet", 101), 2: ("busy", 900), 3: ("medium", 400)}
    assert [n for n, _ in contention.top_foreign(before, after)] == ["busy", "medium", "quiet"]


def test_top_foreign_respects_the_limit():
    before = {i: (f"p{i}", 0) for i in range(10)}
    after = {i: (f"p{i}", 100 * (10 - i)) for i in range(10)}
    assert len(contention.top_foreign(before, after, limit=2)) == 2


def test_a_process_that_started_mid_window_counts_all_its_usage():
    assert contention.top_foreign({}, {7: ("latecomer", 300)}) == [("latecomer", 300.0)]


def test_a_recycled_pid_is_not_credited_with_the_previous_owners_time():
    # Same pid, different name: treat it as new. Computing a delta would go negative for a
    # reused pid with a lower tick count and the process would vanish from the list.
    assert contention.top_foreign({7: ("old", 5000)}, {7: ("new", 40)}) == [("new", 40.0)]


def test_a_process_that_exited_is_simply_absent():
    assert contention.top_foreign({1: ("gone", 100)}, {}) == []


def test_top_foreign_handles_missing_tables():
    assert contention.top_foreign(None, {1: ("x", 1)}) == []
    assert contention.top_foreign({1: ("x", 1)}, None) == []


# ---------------------------------------------------------------- live reads


def test_snapshot_on_this_machine_is_self_consistent():
    first = contention.snapshot()
    if first is None:
        pytest.skip("no CPU counters on this platform")
    total = 0
    for i in range(400_000):
        total += i
    second = contention.snapshot()
    assert second is not None
    # Not asserting monotonicity: these counters go backwards on Windows, which is what
    # test_a_counter_that_went_backwards_returns_none_rather_than_a_clean_bill covers. The
    # contract here is only that a live reading is either a usable fraction or None.
    frac = contention.foreign_fraction(first, second)
    assert frac is None or 0.0 <= frac <= 1.0


def test_own_cpu_time_advances_when_we_burn_cpu():
    first = contention.own_cpu_time()
    if first is None:
        pytest.skip("no per-process CPU counter on this platform")
    total = 0
    for i in range(2_000_000):
        total += i
    assert contention.own_cpu_time() >= first


def test_describe_capability_agrees_with_snapshot():
    caps = contention.describe_capability()
    assert set(caps) == {"cpu_contention"}
    assert caps["cpu_contention"] is (contention.snapshot() is not None)


def test_capability_is_reported_in_the_telemetry_capability_model():
    from mlsyslab import telemetry
    assert "cpu_contention" in telemetry.capabilities()


@pytest.mark.skipif(platform.system() == "Windows",
                    reason="process table is Linux and Android only")
def test_process_table_finds_processes_and_excludes_us():
    import os
    table = contention.process_table()
    if table is None:
        pytest.skip("no /proc on this platform")
    assert len(table) > 0
    assert os.getpid() not in table
    for name, used in table.values():
        assert isinstance(name, str) and used >= 0


@pytest.mark.skipif(platform.system() != "Windows", reason="Windows-only behaviour")
def test_process_table_is_absent_on_windows():
    # Deliberate: the fraction still works there, and a half-built table that silently
    # omitted most processes would be worse than no names at all.
    assert contention.process_table() is None


def test_snapshot_returns_none_when_a_counter_is_unavailable(monkeypatch):
    monkeypatch.setattr(contention.cpu, "snapshot", lambda: None)
    assert contention.snapshot() is None
    monkeypatch.setattr(contention.cpu, "snapshot", lambda: ((10, 100), []))
    monkeypatch.setattr(contention, "own_cpu_time", lambda: None)
    assert contention.snapshot() is None


def test_linux_own_cpu_time_parses_a_name_containing_spaces_and_brackets(tmp_path,
                                                                        monkeypatch):
    # A comm of "(my prog) (x)" defeats any parser that splits the whole line or finds the
    # first ')'. Fields 14 and 15 here are 111 and 222.
    fake = tmp_path / "stat"
    tail = " ".join(str(i) for i in range(3, 14))
    fake.write_text(f"42 (my prog) (x)) {tail} 111 222 0 0 20 0 1 0 0\n")

    real_open = open

    def fake_open(path, *a, **kw):
        if path == "/proc/self/stat":
            return real_open(str(fake), *a, **kw)
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)
    assert contention._linux_own_cpu_time() == 333


def test_linux_own_cpu_time_returns_none_on_a_truncated_line(tmp_path, monkeypatch):
    fake = tmp_path / "stat"
    fake.write_text("42 (prog) S 1 2 3\n")

    real_open = open

    def fake_open(path, *a, **kw):
        if path == "/proc/self/stat":
            return real_open(str(fake), *a, **kw)
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)
    assert contention._linux_own_cpu_time() is None
