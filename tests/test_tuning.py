"""Classical tuning: it must identify the plant, or refuse."""

import pytest
import torch

from entities.systems import Thermal, Trolley
from utils import tuning
from utils.tuning import FOPDT, IdentificationError


def thermal(C=1000.0, h=10.0) -> Thermal:
    return Thermal(
        thermal_capacity=torch.tensor(C),
        heat_transfer_coefficient=torch.tensor(h),
        dt=torch.tensor(0.5),
        initial_temperature=torch.tensor(293.15),
        ambient_temperature=torch.tensor(293.15),
    )


def trolley(friction=0.5) -> Trolley:
    return Trolley(
        mass=torch.tensor(1.0),
        spring=torch.tensor(1.0),
        friction=torch.tensor(friction),
        dt=torch.tensor(0.05),
    )


# ── identification ───────────────────────────────────────────────────────
def test_fopdt_recovers_the_thermal_plants_true_parameters():
    """K = 1/h and T = C/h for this plant, so the fit has a known answer."""
    system = thermal(C=1000.0, h=10.0)
    model = tuning.identify_fopdt(system, steps=2000, final_input=1000.0)
    assert model.K == pytest.approx(1.0 / 10.0, rel=0.05)
    assert model.T == pytest.approx(100.0, rel=0.15)
    assert model.L == pytest.approx(0.0, abs=2.0)


def test_fopdt_refuses_an_oscillatory_plant():
    """The two-point method reads the first overshoot as a monotone approach.

    It does not merely lose accuracy here - it returns a meaningless model, so
    refusing is the correct behaviour.
    """
    with pytest.raises(IdentificationError, match="oscillatory"):
        tuning.identify_fopdt(trolley(friction=0.2), steps=2000, final_input=10.0)


def test_fopdt_refuses_a_zero_step():
    with pytest.raises(IdentificationError, match="No step change"):
        tuning.identify_fopdt(thermal(), steps=100, final_input=0.0, initial_input=0.0)


# ── rules ────────────────────────────────────────────────────────────────
def test_ziegler_nichols_returns_a_derivative_gain_not_a_derivative_time():
    """Kd = Kp * Td. The old code returned 0.5*L, i.e. Td, short by a factor Kp."""
    model = FOPDT(K=0.1, L=2.0, T=100.0)
    Kp, Ki, Kd = tuning.ziegler_nichols(model)
    assert Kp == pytest.approx(1.2 * model.T / (model.K * model.L))
    assert Ki == pytest.approx(Kp / (2.0 * model.L))
    assert Kd == pytest.approx(Kp * 0.5 * model.L)
    assert Kd != pytest.approx(0.5 * model.L)


@pytest.mark.parametrize("rule", [tuning.ziegler_nichols, tuning.cohen_coon])
def test_dead_time_rules_refuse_a_plant_without_dead_time(rule):
    with pytest.raises(IdentificationError, match="dead time"):
        rule(FOPDT(K=0.1, L=0.0, T=100.0))


def test_imc_stays_finite_without_dead_time():
    """L=0 would divide by zero unless lambda is clamped away from it."""
    Kp, Ki, Kd = tuning.pid_imc(FOPDT(K=0.1, L=0.0, T=100.0))
    assert all(map(lambda g: g == g and abs(g) < 1e6, (Kp, Ki, Kd)))
    assert Kd == 0.0  # no derivative action needed on a first-order plant


# ── pole placement ───────────────────────────────────────────────────────
def test_pole_placement_matches_the_analytic_solution_for_the_trolley():
    system = trolley(friction=0.5)
    m, k, c, zeta, bandwidth = 1.0, 1.0, 0.5, 0.8, 2.0
    omega = bandwidth * (k / m) ** 0.5
    Kp, Ki, Kd = tuning.pole_placement(system, bandwidth, zeta)
    assert Kp == pytest.approx(m * (1 + 2 * zeta) * omega**2 - k)
    assert Ki == pytest.approx(m * omega**3)
    assert Kd == pytest.approx(m * (1 + 2 * zeta) * omega - c)


def test_pole_placement_gives_the_thermal_plant_no_derivative_action():
    Kp, Ki, Kd = tuning.pole_placement(thermal())
    assert Kd == 0.0
    assert Kp > 0 and Ki > 0


# ── dispatch ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("build", [thermal, trolley])
def test_auto_finds_a_valid_method_for_either_plant(build):
    """Each plant defeats a different method; "auto" must route around that."""
    Kp, Ki, Kd = tuning.tune(build(), "auto", steps=1500, step_input=10.0)
    assert Kp > 0
    assert all(g == g and g >= 0 for g in (Kp, Ki, Kd))


def test_relay_is_rejected_on_a_plant_with_no_finite_ultimate_gain():
    """Two poles and no dead time never reach -180 degrees of phase.

    A relay run still produces *a* number - it locks onto a cycle at the sample
    rate - so the failure is silent unless something checks. Here the plant is
    slow enough that no limit cycle forms at all within the horizon.
    """
    with pytest.raises(IdentificationError):
        tuning.relay_autotune(thermal(), relay_amplitude=100.0, steps=200)
