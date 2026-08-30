"""Regressions for the defects a code review found in the rewrite itself.

Grouped here rather than scattered because they share a moral: each one was
introduced while fixing something else, and none of them raised.
"""

import subprocess

import numpy as np
import pytest
import torch

from classes.simulation import SimulationConfig, SimulationResults
from comparisons import baselines
from config import load_config
from entities.pid import PID
from learning.scenarios import build_system
from utils import tuning
from utils.run import tracking_loss

DT = torch.tensor(0.05)


# ── the tests were not in the repository ─────────────────────────────────
def test_the_test_suite_is_actually_committed():
    """`.gitignore` carried `test*`, so tests/ was never tracked and CI ran
    against an empty suite, exiting 5 with "no tests ran"."""
    tracked = subprocess.run(
        ["git", "ls-files", "tests/"], capture_output=True, text=True, check=True
    ).stdout.split()
    assert any(name.endswith(".py") for name in tracked), (
        "no test file is tracked by git — check .gitignore"
    )
    assert "tests/test_review_findings.py" in tracked


# ── the classical baseline was crippled by its own measurement ───────────
@pytest.mark.parametrize("system_name", ["trolley", "thermal"])
def test_step_test_is_long_enough_to_identify_the_plant(system_name):
    """A fixed 600 samples is 3 tau on the thermal plant: the response has not
    settled, so K and T come out low and a dead time appears that is not there,
    which sent IMC to Kp=227, Kd=553 instead of Kp=100, Kd=0."""
    config = load_config(system_name)
    system = build_system(system_name, config)
    steps = tuning.step_test_steps(system)

    _, response = system.step_response(steps=steps, final_input=1.0)
    tail = response[int(0.9 * len(response)) :]
    final = float(response[-1])
    assert abs(float(tail.max()) - float(tail.min())) < 0.02 * abs(final - float(response[0])), (
        "step response has not settled by the end of the test window"
    )


def test_thermal_classical_tuning_recovers_a_pi_controller():
    config = load_config("thermal")
    system = build_system("thermal", config)
    Kp, Ki, Kd = baselines.classical(system, config).gains
    assert Kp == pytest.approx(100.0, rel=0.1)
    assert Kd == 0.0, "a first-order plant with no dead time needs no derivative"


@pytest.mark.parametrize("system_name", ["trolley", "thermal"])
def test_configured_initial_gains_match_what_the_tuner_returns(system_name):
    """The warm start is documented as starting at the classical gains, which is
    only true if the config and the tuner agree."""
    config = load_config(system_name)
    system = build_system(system_name, config)
    tuned = baselines.classical(system, config).gains
    for name, configured, actual in zip(
        ("Kp", "Ki", "Kd"), config.control.initial_gains, tuned, strict=True
    ):
        assert configured == pytest.approx(actual, rel=0.05, abs=0.05), (
            f"{system_name} {name}: config says {configured}, tuner says {actual}"
        )


# ── the loss had the same sign bug the metrics used to have ──────────────
def _window(predicted, reference):
    results = SimulationResults()
    results.positions = [torch.tensor(float(v)) for v in predicted]
    results.setpoints = [torch.tensor(float(v)) for v in reference]
    config = SimulationConfig(setpoints=results.setpoints, dt=0.05, error_scale=10.0)
    return tracking_loss(results, config, 0, len(predicted), overshoot_weight=1.0)


def test_overshoot_penalty_is_symmetric_under_reversing_the_step():
    """relu(y - setpoint) only penalises overshoot on a rising step. On a falling
    one it rewards it, biasing the optimum to an offset on the near side."""
    rising = _window([0, 5, 12, 10, 10], [10] * 5)     # overshoots to 12
    falling = _window([0, -5, -12, -10, -10], [-10] * 5)  # mirror image
    assert float(rising) == pytest.approx(float(falling), rel=1e-5)


def test_overshoot_costs_more_than_an_equal_undershoot():
    over = _window([0, 5, 12, 10, 10], [10] * 5)
    under = _window([0, 5, 8, 10, 10], [10] * 5)
    assert float(over) > float(under)


# ── PID state and anti-windup ────────────────────────────────────────────
def test_dropping_the_measurement_restores_derivative_on_error():
    """A stale _y_k left the derivative looking enabled with no history, so the
    term returned a hard zero forever."""
    pid = PID(torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0))
    for y in (0.0, 0.1, 0.2):
        pid.compute(torch.tensor(1.0 - y), DT, measurement=torch.tensor(y))
    assert pid._y_k is not None

    pid.compute(torch.tensor(1.0), DT)  # measurement dropped
    assert pid._y_k is None and pid.y_k_1 is None and pid.y_k_2 is None

    outputs = [float(pid.compute(torch.tensor(1.0 - 0.3 * i), DT)) for i in range(4)]
    assert len(set(round(o, 6) for o in outputs)) > 1, (
        "derivative term is stuck — the error-based fallback did not engage"
    )


def test_anti_windup_works_on_an_actuator_that_cannot_go_negative():
    """The old test was sign(u)*sign(e), which is only right for a range that
    straddles zero. A heater is limited to [0, P_max]."""
    pid = PID(torch.tensor(1.0), torch.tensor(50.0), torch.tensor(0.0))
    pid.set_limits(torch.tensor(4000.0), torch.tensor(0.0))

    for _ in range(200):  # drive hard into the upper limit
        pid.compute_backward_euler(torch.tensor(50.0), DT)
    wound = float(pid.integral)

    for _ in range(200):
        pid.compute_backward_euler(torch.tensor(50.0), DT)
    assert float(pid.integral) == pytest.approx(wound, rel=1e-6), "integral wound up"

    # The error flips: the controller must come off the limit promptly.
    outputs = [float(pid.compute_backward_euler(torch.tensor(-50.0), DT)) for _ in range(3)]
    assert outputs[0] < 4000.0


# ── the recurrence was inert ─────────────────────────────────────────────
def test_hidden_state_changes_the_prediction_at_the_configured_window():
    """With a 40-sample window the carried hidden state moved the gains by 0.0:
    the window already held everything it could remember, at 40x the cost."""
    from models.pid_lstm import LSTMAdaptivePID

    config = load_config("trolley")
    length = config.learning.lstm.sequence_length
    torch.manual_seed(0)
    model = LSTMAdaptivePID(5, 32, 3)
    model.eval()

    hidden = None
    with torch.no_grad():
        for _ in range(30):
            x = torch.randn(1, length, 5) * 0.3
            with_state, hidden = model(x, hidden)
            without_state, _ = model(x, None)
    difference = float((with_state - without_state).abs().max())
    assert difference > 1e-5, (
        f"hidden state contributes {difference:.1e} at sequence_length={length}; "
        "the recurrence is doing nothing and the window is redundant work"
    )


# ── fail fast ────────────────────────────────────────────────────────────
def test_training_without_a_network_fails_before_simulating():
    from learning.utils import extract_lstm_input, extract_rbf_input
    from models.sys_rbf import SystemRBFModel
    from utils.run import run_episode

    config = load_config("trolley")
    system = build_system("trolley", config)
    pid = PID(torch.tensor(1.0), torch.tensor(0.0), torch.tensor(0.0))
    rbf = SystemRBFModel(
        input_mean=torch.zeros(4), input_std=torch.ones(4),
        output_mean=torch.zeros(1), output_std=torch.ones(1),
        hidden_features=4, input_size=4,
    )
    simulation_config = SimulationConfig(
        setpoints=[torch.tensor(1.0)] * 50, dt=torch.tensor(config.learning.dt)
    )
    with pytest.raises(ValueError, match="needs an LSTM model"):
        run_episode(
            system=system, pid=pid, simulation_config=simulation_config,
            extract_rbf_input=extract_rbf_input.trolley,
            extract_lstm_input=extract_lstm_input,
            rbf_model=rbf, lstm_model=None, session="train",
            optimizer=torch.optim.SGD(rbf.parameters(), lr=0.0),
        )
