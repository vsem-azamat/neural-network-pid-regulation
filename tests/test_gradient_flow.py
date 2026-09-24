"""The regression test for this project's central defect.

The original training loop rebuilt the RBF input with ``torch.tensor([...])``,
which copies numbers out of the autograd graph. The loss still had
``requires_grad=True`` (the RBF's own weights were in the graph), so nothing
raised — but every LSTM parameter came back with ``grad=None`` and ten epochs of
training changed nothing. These tests fail loudly if that ever comes back.
"""

import pytest
import torch

from classes.simulation import SimulationConfig
from entities.pid import PID
from entities.systems import Trolley
from learning.utils import N_FEATURES, extract_lstm_input, extract_rbf_input
from models.pid_lstm import LSTMAdaptivePID
from models.sys_rbf import SystemRBFModel
from utils.run import run_episode


def build(steps: int = 60):
    torch.manual_seed(0)
    system = Trolley(
        mass=torch.tensor(1.0),
        spring=torch.tensor(1.0),
        friction=torch.tensor(0.5),
        dt=torch.tensor(0.05),
    )
    pid = PID(torch.tensor(5.0), torch.tensor(0.5), torch.tensor(1.0))
    pid.set_limits(torch.tensor(50.0), torch.tensor(-50.0))
    lstm = LSTMAdaptivePID(input_size=N_FEATURES, hidden_size=16, output_size=3)
    rbf = SystemRBFModel(
        input_mean=torch.zeros(4),
        input_std=torch.ones(4),
        output_mean=torch.zeros(1),
        output_std=torch.ones(1),
        hidden_features=12,
        input_size=4,
    )
    config = SimulationConfig(
        setpoints=[torch.tensor(5.0)] * steps,
        dt=torch.tensor(0.05),
        sequence_length=10,
        tbptt_window=10,
        warm_up_steps=5,
        pid_gain_factor=(20.0, 2.0, 5.0),
        error_scale=10.0,
        operating_range=(-10.0, 10.0),
    )
    return system, pid, lstm, rbf, config


def run(session="train", optimizer=None, lstm=None, **kw):
    system, pid, default_lstm, rbf, config = build()
    lstm = default_lstm if lstm is None else lstm
    return run_episode(
        system=system,
        pid=pid,
        simulation_config=config,
        extract_rbf_input=extract_rbf_input.trolley,
        extract_lstm_input=extract_lstm_input,
        rbf_model=rbf,
        lstm_model=lstm if session != "static" else None,
        session=session,
        optimizer=optimizer,
        **kw,
    )


def test_gradient_reaches_every_lstm_parameter():
    system, pid, lstm, rbf, config = build()
    optimizer = torch.optim.SGD(lstm.parameters(), lr=1e-3)
    report = run_episode(
        system=system,
        pid=pid,
        simulation_config=config,
        extract_rbf_input=extract_rbf_input.trolley,
        extract_lstm_input=extract_lstm_input,
        rbf_model=rbf,
        lstm_model=lstm,
        session="train",
        optimizer=optimizer,
    )

    assert report.grad_norms, "no TBPTT window ever fired"
    assert all(g > 0 for g in report.grad_norms), (
        f"gradient vanished in some window: {report.grad_norms}"
    )
    for name, param in lstm.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradient"


def test_optimizer_actually_changes_weights():
    system, pid, lstm, rbf, config = build()
    before = [p.detach().clone() for p in lstm.parameters()]
    optimizer = torch.optim.SGD(lstm.parameters(), lr=0.1)
    run_episode(
        system=system,
        pid=pid,
        simulation_config=config,
        extract_rbf_input=extract_rbf_input.trolley,
        extract_lstm_input=extract_lstm_input,
        rbf_model=rbf,
        lstm_model=lstm,
        session="train",
        optimizer=optimizer,
    )
    after = list(lstm.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True)), (
        "training completed without changing a single weight"
    )


def test_rbf_input_stays_attached_to_the_graph():
    """The exact failure: torch.tensor([...]) would make this requires_grad=False."""
    system = Trolley(
        mass=torch.tensor(1.0),
        spring=torch.tensor(1.0),
        friction=torch.tensor(0.5),
        dt=torch.tensor(0.05),
    )
    control = torch.tensor(3.0, requires_grad=True)
    system.apply_control(control)
    row = extract_rbf_input.trolley(system, control)
    assert row.requires_grad, "RBF input was detached from the control signal"
    row.sum().backward()
    assert control.grad is not None and control.grad.abs() > 0


def test_validation_and_static_sessions_build_no_graph():
    for session in ("validation", "static"):
        report = run(session=session)
        assert not report.grad_norms
        assert not report.results.positions[-1].requires_grad


def test_training_requires_an_optimizer():
    system, pid, lstm, rbf, config = build()
    with pytest.raises(ValueError, match="Optimizer"):
        run_episode(
            system=system,
            pid=pid,
            simulation_config=config,
            extract_rbf_input=extract_rbf_input.trolley,
            extract_lstm_input=extract_lstm_input,
            rbf_model=rbf,
            lstm_model=lstm,
            session="train",
        )
