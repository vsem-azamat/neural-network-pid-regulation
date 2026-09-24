"""Regression tests for the three step-response metrics that were wrong.

Each test below is a case the previous implementation got wrong, not a
hypothetical. The trolley comparison drew setpoints from (-20, 20), so roughly
half of every reported run hit the negative-setpoint bugs.
"""

import numpy as np
import pytest

from utils.metrics import aggregate, compute_metrics

DT = 0.1


def damped_oscillation(setpoint: float, duration: float = 20.0) -> np.ndarray:
    t = np.arange(0, duration, DT)
    return setpoint * (1 - np.exp(-0.5 * t) * np.cos(1.5 * t))


def test_overshoot_is_sign_agnostic():
    """max(y) - setpoint reported 100% for any negative setpoint."""
    positive = compute_metrics(damped_oscillation(10.0), 10.0, DT, initial_value=0.0)
    negative = compute_metrics(damped_oscillation(-10.0), -10.0, DT, initial_value=0.0)
    assert positive.overshoot == pytest.approx(negative.overshoot, rel=1e-6)
    assert 30.0 < negative.overshoot < 45.0  # ~37% for this response


def test_rise_time_is_sign_agnostic():
    """`p >= 0.9 * abs(setpoint)` is never true on a negative step."""
    positive = compute_metrics(damped_oscillation(10.0), 10.0, DT, initial_value=0.0)
    negative = compute_metrics(damped_oscillation(-10.0), -10.0, DT, initial_value=0.0)
    assert positive.rise_time == pytest.approx(negative.rise_time, rel=1e-6)
    assert negative.rise_time < 5.0  # not the full simulation length


def test_settling_time_rejects_a_response_that_never_settles():
    """Entering the band once is not settling; the old version reported 2.9 s."""
    y = np.concatenate(
        [np.linspace(0, 10.0, 30), 10 + 5 * np.sin(np.arange(170) * 0.3)]
    )
    metrics = compute_metrics(y, 10.0, DT, initial_value=0.0)
    assert np.isnan(metrics.settling_time)


def test_settling_time_is_the_last_exit_from_the_band():
    """A response that leaves the band and comes back settles at the later time."""
    y = np.concatenate([
        np.full(20, 10.0),   # inside the band immediately
        np.full(20, 12.0),   # leaves it
        np.full(60, 10.0),   # returns for good
    ])
    metrics = compute_metrics(y, 10.0, DT, initial_value=0.0)
    assert metrics.settling_time == pytest.approx(40 * DT, abs=DT)


def test_matches_textbook_first_order_response():
    """Both times land on the first sample past the analytic threshold."""
    t = np.arange(0, 20, DT)
    y = 10 * (1 - np.exp(-t))  # tau = 1 s
    metrics = compute_metrics(y, 10.0, DT, initial_value=0.0)
    assert metrics.overshoot == 0.0
    assert metrics.rise_time == pytest.approx(2.30, abs=1.5 * DT)      # ln(10) tau
    assert metrics.settling_time == pytest.approx(3.91, abs=1.5 * DT)  # ln(50) tau


def test_step_downwards_from_a_nonzero_start():
    """Travel is measured from the initial value, not from zero."""
    y = 100 - 40 * (1 - np.exp(-np.arange(0, 20, DT)))
    metrics = compute_metrics(y, 60.0, DT, initial_value=100.0)
    assert metrics.overshoot == 0.0
    assert metrics.rise_time == pytest.approx(2.30, abs=1.5 * DT)


def test_aggregate_reports_how_many_runs_produced_a_number():
    settling = compute_metrics(np.full(50, 5.0), 5.0, DT, initial_value=5.0)
    never = compute_metrics(
        np.concatenate([np.linspace(0, 5, 10), 5 + 3 * np.sin(np.arange(40))]),
        5.0, DT, initial_value=0.0,
    )
    summary = aggregate([settling, never, settling])
    assert summary["settling_time"]["n_valid"] == 2
    assert summary["settling_time"]["n_total"] == 3
    assert not np.isnan(summary["settling_time"]["mean"])
