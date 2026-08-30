"""PID behaviour, including the two defects that dominated the step responses."""

import pytest
import torch

from entities.pid import PID

DT = torch.tensor(0.05)


def pid(kp=2.0, ki=0.5, kd=1.0, limits=(50.0, -50.0)) -> PID:
    controller = PID(torch.tensor(kp), torch.tensor(ki), torch.tensor(kd))
    if limits:
        controller.set_limits(torch.tensor(limits[0]), torch.tensor(limits[1]))
    return controller


# ── derivative kick ──────────────────────────────────────────────────────
def test_derivative_on_measurement_removes_the_setpoint_kick():
    """A step in the reference is a discontinuity in the error.

    With Kd=1 and dt=0.05, differentiating it asks for 200 units of control on
    a single sample from a 10-unit setpoint change. Differentiating the
    (unchanged) measurement instead asks for none.
    """
    on_error = pid(kd=1.0, limits=None)
    on_measurement = pid(kd=1.0, limits=None)

    measurement = torch.tensor(0.0)
    kick = on_error.compute(torch.tensor(10.0), DT)
    calm = on_measurement.compute(torch.tensor(10.0), DT, measurement=measurement)

    assert abs(float(kick)) > 100.0
    assert abs(float(calm)) < 25.0


def test_derivative_terms_agree_once_the_reference_is_constant():
    """The two formulations differ only where the reference moves.

    Compared in a positional form: the incremental form carries u_{k-1}
    forward, so the initial kick stays in the output as a constant offset
    forever and the two would never line up again.
    """
    on_error, on_measurement = pid(limits=None), pid(limits=None)
    setpoint, y = 5.0, 0.0
    for step in range(20):
        y += 0.1  # a plant response, independent of the controller
        error = torch.tensor(setpoint - y)
        a = on_error.compute(error, DT, method="backward_euler")
        b = on_measurement.compute(
            error, DT, method="backward_euler", measurement=torch.tensor(y)
        )
        if step > 2:  # after both have filled their history
            assert float(a) == pytest.approx(float(b), rel=1e-4)


# ── anti-windup ──────────────────────────────────────────────────────────
def test_incremental_form_stores_the_saturated_output():
    """u_{k-1} is the integrator here, so an unclamped value is windup."""
    controller = pid(kp=50.0, ki=50.0, limits=(10.0, -10.0))
    for _ in range(50):
        controller.compute(torch.tensor(100.0), DT)
    assert float(controller.u_k_1) <= 10.0 + 1e-6


def test_saturated_controller_recovers_immediately_when_the_error_flips():
    """With windup the output stays pinned long after the error changes sign."""
    controller = pid(kp=5.0, ki=20.0, limits=(10.0, -10.0))
    for _ in range(200):
        controller.compute(torch.tensor(50.0), DT)   # drive hard into the limit
    assert float(controller.compute(torch.tensor(50.0), DT)) == pytest.approx(10.0)

    outputs = [float(controller.compute(torch.tensor(-50.0), DT)) for _ in range(5)]
    assert outputs[0] < 10.0, "output did not respond to the error changing sign"


def test_positional_form_stops_integrating_while_saturated():
    controller = pid(kp=1.0, ki=10.0, limits=(5.0, -5.0))
    for _ in range(100):
        controller.compute_backward_euler(torch.tensor(20.0), DT)
    bounded = float(controller.integral)
    for _ in range(100):
        controller.compute_backward_euler(torch.tensor(20.0), DT)
    assert float(controller.integral) == pytest.approx(bounded, rel=1e-6)


# ── plumbing ─────────────────────────────────────────────────────────────
def test_output_respects_the_limits():
    controller = pid(kp=1000.0, limits=(3.0, -3.0))
    assert float(controller.compute(torch.tensor(100.0), DT)) == pytest.approx(3.0)
    controller.reset()
    assert float(controller.compute(torch.tensor(-100.0), DT)) == pytest.approx(-3.0)


def test_gradient_flows_from_the_output_back_to_the_gains():
    kp = torch.tensor(2.0, requires_grad=True)
    controller = PID(kp, torch.tensor(0.5), torch.tensor(0.0))
    controller.compute(torch.tensor(1.0), DT).backward()
    assert kp.grad is not None and float(kp.grad) != 0.0


def test_detach_state_keeps_the_values():
    controller = pid()
    controller.compute(torch.tensor(1.0), DT, measurement=torch.tensor(0.5))
    before = float(controller.u_k_1)
    controller.detach_state()
    assert not controller.u_k_1.requires_grad
    assert float(controller.u_k_1) == pytest.approx(before)


def test_reset_clears_every_piece_of_state():
    controller = pid()
    for _ in range(10):
        controller.compute(torch.tensor(3.0), DT, measurement=torch.tensor(1.0))
    controller.reset()
    assert float(controller.integral) == 0.0
    assert float(controller.u_k_1) == 0.0
    assert controller._y_k is None


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown method"):
        pid().compute(torch.tensor(1.0), DT, method="quantum")  # type: ignore[arg-type]
