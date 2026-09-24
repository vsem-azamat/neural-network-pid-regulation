"""Pins for the residual-scheduling architecture.

The first trained scheduler emitted absolute gains and landed below one
constant gain triple (captured headroom -8.6 % of the +9.7 % available).
These tests pin the three fixes: a residual head centred on the baseline
controller, a penalty on the rate of gain change, and a saturation feature.
"""

import torch

from classes.simulation import SimulationConfig, SimulationResults
from entities.pid import PID
from entities.systems import Trolley
from learning.utils import N_FEATURES, extract_lstm_input, extract_rbf_input
from models.pid_lstm import LSTMAdaptivePID
from utils.run import run_episode, tracking_loss

BASELINE = (40.0, 100.0, 9.0)
CEILING = (150.0, 200.0, 50.0)


def make_config(steps: int = 40) -> SimulationConfig:
    return SimulationConfig(
        setpoints=[torch.tensor(5.0)] * steps,
        dt=torch.tensor(0.05),
        sequence_length=1,
        tbptt_window=10,
        warm_up_steps=5,
        pid_gain_factor=CEILING,
        error_scale=10.0,
        operating_range=(-10.0, 10.0),
        control_scale=50.0,
    )


# ── residual head ────────────────────────────────────────────────────────
def test_untrained_residual_scheduler_is_exactly_its_baseline():
    """The whole point of the residual form: before training, the scheduler
    IS the best-constant controller it will be compared against."""
    torch.manual_seed(0)
    model = LSTMAdaptivePID(
        N_FEATURES, 16, 3, baseline_gains=BASELINE, gain_ceiling=CEILING
    )
    fractions, _ = model(torch.randn(1, 1, N_FEATURES))
    gains = fractions[0] * torch.tensor(CEILING)
    assert torch.allclose(gains, torch.tensor(BASELINE), atol=1e-5), (
        f"untrained residual scheduler emits {gains.tolist()}, "
        f"not its baseline {BASELINE}"
    )


def test_residual_correction_is_bounded_and_clamped_to_the_ceiling():
    torch.manual_seed(0)
    model = LSTMAdaptivePID(
        N_FEATURES, 16, 3,
        baseline_gains=(100.0, 150.0, 40.0),
        gain_ceiling=CEILING,
        residual_range=2.5,
    )
    # Saturate the head both ways by brute force on the bias.
    for sign in (+1.0, -1.0):
        with torch.no_grad():
            model.linear.bias.fill_(sign * 50.0)
        fractions, _ = model(torch.zeros(1, 1, N_FEATURES))
        gains = fractions[0] * torch.tensor(CEILING)
        assert bool((gains <= torch.tensor(CEILING) + 1e-4).all()), (
            f"gains {gains.tolist()} escape the ceiling {CEILING}"
        )
        if sign < 0:
            expected_floor = torch.tensor((100.0, 150.0, 40.0)) / 2.5
            assert torch.allclose(gains, expected_floor, rtol=1e-4), (
                "the lower edge of the correction band should be baseline/range"
            )


def test_the_checkpoint_carries_the_residual_mapping():
    """Evaluation code builds the model from architecture arguments only and
    loads the state dict; the baseline must survive that round trip."""
    model = LSTMAdaptivePID(
        N_FEATURES, 16, 3, baseline_gains=BASELINE, gain_ceiling=CEILING
    )
    fresh = LSTMAdaptivePID(N_FEATURES, 16, 3)
    assert not fresh.residual_mode
    fresh.load_state_dict(model.state_dict())
    assert fresh.residual_mode
    assert fresh.warm_start_gains == BASELINE


def test_gradient_reaches_the_weights_despite_zero_init():
    model = LSTMAdaptivePID(
        N_FEATURES, 16, 3, baseline_gains=BASELINE, gain_ceiling=CEILING
    )
    fractions, _ = model(torch.randn(1, 1, N_FEATURES))
    fractions.sum().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0.0, (
        "zero-initialised head must still pass gradient to the LSTM below it"
    )


# ── gain-rate penalty ────────────────────────────────────────────────────
def _results_with_gain_trace(kp_trace: list[float]) -> SimulationResults:
    steps = len(kp_trace)
    results = SimulationResults()
    results.setpoints = [torch.tensor(5.0)] * steps
    results.positions = [torch.tensor(5.0)] * steps  # zero tracking error
    results.kp_values = [torch.tensor(k) for k in kp_trace]
    results.ki_values = [torch.tensor(100.0)] * steps
    results.kd_values = [torch.tensor(9.0)] * steps
    return results


def test_gain_chatter_is_penalised_and_a_smooth_schedule_is_not():
    config = make_config(steps=20)
    chatter = _results_with_gain_trace([40.0, 80.0] * 10)
    smooth = _results_with_gain_trace([40.0 + i * 0.5 for i in range(20)])
    kwargs = dict(window_start=0, window_end=20, gain_rate_weight=0.02)
    chatter_loss = float(tracking_loss(chatter, config, **kwargs))
    smooth_loss = float(tracking_loss(smooth, config, **kwargs))
    assert chatter_loss > 10 * smooth_loss, (
        f"chattering gains cost {chatter_loss:.4f} vs {smooth_loss:.4f} smooth; "
        "the rate penalty is not doing its job"
    )
    without = float(tracking_loss(chatter, config, window_start=0, window_end=20))
    assert without < chatter_loss, "weight 0 must disable the penalty"


# ── saturation feature ───────────────────────────────────────────────────
def test_the_network_sees_the_control_signal():
    config = make_config()
    config.sequence_length = 3
    results = SimulationResults()
    for value in (10.0, 25.0, -50.0):
        results.error_history.append(torch.tensor(0.1))
        results.error_diff_history.append(torch.tensor(0.0))
        results.kp_values.append(torch.tensor(40.0))
        results.ki_values.append(torch.tensor(100.0))
        results.kd_values.append(torch.tensor(9.0))
        results.positions.append(torch.tensor(1.0))
        results.setpoints.append(torch.tensor(5.0))
        results.control_outputs.append(torch.tensor(value))
    window = extract_lstm_input(config, results)
    assert window.shape == (1, 3, N_FEATURES)
    expected = torch.tensor([10.0, 25.0, -50.0]) / config.control_scale
    assert torch.allclose(window[0, :, 7], expected), (
        "feature 7 must be the control signal over the actuator range"
    )


# ── optional surrogate ───────────────────────────────────────────────────
def test_an_episode_runs_without_the_rbf_surrogate():
    system = Trolley(
        mass=torch.tensor(1.0),
        spring=torch.tensor(1.0),
        friction=torch.tensor(0.5),
        dt=torch.tensor(0.05),
    )
    pid = PID(torch.tensor(5.0), torch.tensor(0.5), torch.tensor(1.0))
    pid.set_limits(torch.tensor(50.0), torch.tensor(-50.0))
    results = run_episode(
        system=system,
        pid=pid,
        simulation_config=make_config(),
        extract_rbf_input=extract_rbf_input.trolley,
        extract_lstm_input=extract_lstm_input,
        rbf_model=None,
        lstm_model=None,
        session="static",
    ).results
    assert len(results.positions) == 40
    assert results.rbf_predictions == []
