"""Physics of the two plants, including the modelling bugs that were fixed."""

import pytest
import torch

from entities.systems import Thermal, Trolley

AMBIENT = 293.15


def thermal(**kwargs) -> Thermal:
    defaults = dict(
        thermal_capacity=torch.tensor(1000.0),
        heat_transfer_coefficient=torch.tensor(10.0),
        dt=torch.tensor(0.5),
        initial_temperature=torch.tensor(AMBIENT),
        ambient_temperature=torch.tensor(AMBIENT),
    )
    return Thermal(**{**defaults, **kwargs})


def trolley(**kwargs) -> Trolley:
    defaults = dict(
        mass=torch.tensor(1.0),
        spring=torch.tensor(1.0),
        friction=torch.tensor(0.5),
        dt=torch.tensor(0.05),
    )
    return Trolley(**{**defaults, **kwargs})


# ── thermal ──────────────────────────────────────────────────────────────
def test_thermal_at_ambient_with_no_heat_stays_put():
    """The old model, C*dT/dt = Q - h*T, cooled towards 0 K instead."""
    system = thermal()
    for _ in range(2000):
        system.apply_control(torch.tensor(0.0))
    assert float(system.X) == pytest.approx(AMBIENT, abs=1e-3)


def test_thermal_cools_towards_ambient_not_absolute_zero():
    system = thermal(initial_temperature=torch.tensor(400.0))
    for _ in range(20_000):
        system.apply_control(torch.tensor(0.0))
    assert float(system.X) == pytest.approx(AMBIENT, abs=0.5)


def test_thermal_steady_state_matches_the_analytic_value():
    """T_ss = T_amb + Q/h."""
    system = thermal()
    power = 500.0
    for _ in range(50_000):
        system.apply_control(torch.tensor(power))
    assert float(system.X) == pytest.approx(AMBIENT + power / 10.0, rel=1e-4)


def test_thermal_steady_state_power_is_what_holds_the_temperature():
    system = thermal(initial_temperature=torch.tensor(350.0))
    holding = system.steady_state_power
    before = float(system.X)
    for _ in range(100):
        system.apply_control(holding)
    assert float(system.X) == pytest.approx(before, abs=1e-3)


def test_thermal_time_constant():
    assert float(thermal().tau) == pytest.approx(100.0)


# ── trolley ──────────────────────────────────────────────────────────────
def test_trolley_settles_at_the_spring_equilibrium():
    """Steady state of m*x'' + c*x' + k*x = F is x = F/k."""
    system = trolley()
    force = 7.0
    for _ in range(20_000):
        system.apply_control(torch.tensor(force))
    assert float(system.X) == pytest.approx(force / 1.0, rel=1e-3)


def test_trolley_damping_ratio():
    # zeta = c / (2*sqrt(k*m))
    assert float(trolley().damping_ratio) == pytest.approx(0.25)
    assert float(trolley(friction=torch.tensor(2.0)).damping_ratio) == pytest.approx(1.0)


def test_underdamped_trolley_overshoots_and_overdamped_does_not():
    peaks = {}
    for name, friction in (("under", 0.2), ("over", 4.0)):
        system = trolley(friction=torch.tensor(friction))
        peak = 0.0
        for _ in range(4000):
            system.apply_control(torch.tensor(5.0))
            peak = max(peak, float(system.X))
        peaks[name] = peak
    assert peaks["under"] > 5.0 * 1.1
    assert peaks["over"] <= 5.0 * 1.001


# ── shared conventions ───────────────────────────────────────────────────
@pytest.mark.parametrize("build", [thermal, trolley])
def test_disturbance_adds_to_the_control_channel_in_both_plants(build):
    """The trolley used to subtract it, making robustness incomparable."""
    with_control = build()
    with_disturbance = build()
    for _ in range(50):
        with_control.apply_control(torch.tensor(10.0))
        with_disturbance.apply_control(torch.tensor(6.0), torch.tensor(4.0))
    assert float(with_control.X) == pytest.approx(float(with_disturbance.X), rel=1e-5)


@pytest.mark.parametrize("build", [thermal, trolley])
def test_min_dt_accepts_an_oversampling_factor(build):
    """It was a @property, so the argument was unreachable and pinned at 10."""
    system = build()
    assert float(system.min_dt(10.0)) > float(system.min_dt(50.0))


@pytest.mark.parametrize("build", [thermal, trolley])
def test_reset_restores_the_initial_state_and_dtype(build):
    system = build()
    initial = float(system.X)
    for _ in range(100):
        system.apply_control(torch.tensor(5.0))
    system.reset()
    assert float(system.X) == pytest.approx(initial)
    assert system.X.dtype == torch.float32  # was int64 for the trolley


@pytest.mark.parametrize("build", [thermal, trolley])
def test_detach_state_keeps_the_value_and_drops_the_history(build):
    system = build()
    system.apply_control(torch.tensor(3.0, requires_grad=True))
    assert system.X.requires_grad
    value = float(system.X.detach())
    system.detach_state()
    assert not system.X.requires_grad
    assert float(system.X) == pytest.approx(value)


@pytest.mark.parametrize("build", [thermal, trolley])
def test_step_response_starts_from_the_pre_step_operating_point(build):
    system = build()
    time, output = system.step_response(steps=50, final_input=5.0)
    assert len(time) == len(output) == 51
    assert float(time[0]) == 0.0
    assert float(output[0]) == pytest.approx(float(build().X))
