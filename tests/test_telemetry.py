import time

from mlsyslab.telemetry import TelemetrySession
from mlsyslab.telemetry.cpu import utilisation
from mlsyslab.telemetry.power import integrate_energy, rapl_delta_j
from mlsyslab.sysinfo import default_thread_sweep


def test_integrate_energy_trapezoidal():
    # 2 W for one second, then 4 W for one second: 2 J + ramp average 3 J = 5 J? No:
    # trapezoid between (0,2) and (1,2) is 2 J; between (1,2) and (2,4) is 3 J.
    samples = [(0.0, 2.0), (1.0, 2.0), (2.0, 4.0)]
    assert integrate_energy(samples) == 5.0


def test_integrate_energy_needs_two_samples():
    assert integrate_energy([(0.0, 2.0)]) is None
    assert integrate_energy([]) is None


def test_integrate_energy_ignores_non_monotonic_time():
    samples = [(0.0, 2.0), (1.0, 2.0), (0.5, 99.0)]  # clock went backwards
    assert integrate_energy(samples) == 2.0


def test_rapl_delta_normal_case():
    assert rapl_delta_j(1_000_000, 3_500_000) == 2.5


def test_cpu_utilisation_from_snapshots():
    # (idle, total): 30 idle out of 100 total elapsed = 70% busy.
    before = ((100, 1000), [(50, 500), (50, 500)])
    after = ((130, 1100), [(80, 550), (50, 550)])
    result = utilisation(before, after)
    assert result["mean"] == 70.0
    assert result["per_core"] == [40.0, 100.0]


def test_cpu_utilisation_clamps_jitter():
    before = ((100, 1000), [])
    after = ((90, 1100), [])  # idle counter jitter would give >100%
    assert 0.0 <= utilisation(before, after)["mean"] <= 100.0


def test_cpu_utilisation_handles_missing_snapshots():
    assert utilisation(None, None) == {"mean": None, "per_core": None}


def test_session_summary_windows_samples():
    session = TelemetrySession(sample_hz=50, sample_power=False)
    session.start()
    time.sleep(0.35)
    session.stop()
    full = session.summary()
    assert full["sample_count"] >= 3
    # A window in the far future contains nothing and reports honestly.
    empty = session.summary(t0=time.perf_counter() + 100)
    assert empty["sample_count"] == 0


def test_session_energy_per_token_division():
    session = TelemetrySession(sample_hz=50, sample_power=False)
    session.start()
    time.sleep(0.1)
    session.stop()
    # Inject a synthetic power trace so the arithmetic is exact.
    t0 = 1000.0
    session._samples = [{"t": t0 + i * 0.1, "w_total": 5.0} for i in range(11)]
    summary = session.summary(t0, t0 + 1.0, tokens=100)
    assert abs(summary["energy_j"] - 5.0) < 1e-9
    assert abs(summary["energy_per_token_mj"] - 50.0) < 1e-6


def test_default_thread_sweep_is_small_and_sorted():
    sweep = default_thread_sweep(max_threads=8)
    assert sweep == sorted(set(sweep))
    assert sweep[0] == 1
    assert sweep[-1] <= 8
    assert len(sweep) <= 4
