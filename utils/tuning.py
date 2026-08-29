"""Classical PID tuning — the baseline the neural controller is measured against.

A comparison is only as meaningful as the baseline it beats, so this module
identifies the plant properly instead of forcing one model onto every system:

* :func:`identify_fopdt` fits a first-order-plus-dead-time model from an
  open-loop step test, by the two-point (Smith) method.  It is the right model
  for the thermal plant, and ``tests/test_tuning.py`` pins it against that
  plant's analytically known K and T.
* :func:`relay_autotune` runs an Åström–Hägglund relay experiment to measure the
  ultimate gain and period directly, for plants that have a finite one.
* :func:`pole_placement` solves for the gains from the plant's own structure.
  It is the right method for the trolley: the two-point fit reads its first
  overshoot as if it were a monotone approach, and the relay method does not
  apply either, because a relative-degree-two plant with no dead time never
  reaches −180° of phase and so has no finite ultimate gain at all — a relay run
  on it locks onto a chattering cycle at the sample rate and reports an
  "ultimate gain" of several hundred, which is an artefact of the sampling.

:func:`tune` picks between them, so callers ask for a method and get a baseline
that is actually valid for their plant. ``method="auto"`` is the one to use.
"""

from dataclasses import dataclass
from typing import Literal

import torch

from entities.systems import BaseSystem

TuningMethod = Literal[
    "ziegler_nichols", "cohen_coon", "pid_imc", "relay", "pole_placement", "auto"
]

Gains = tuple[float, float, float]


@dataclass(frozen=True)
class FOPDT:
    """First-order-plus-dead-time process model: K·e^(−L·s) / (T·s + 1)."""

    K: float  # process gain
    L: float  # dead time, s
    T: float  # time constant, s


class IdentificationError(RuntimeError):
    """The step test did not produce a response the method can interpret."""


# ── process identification ───────────────────────────────────────────────
def identify_fopdt(
    system: BaseSystem,
    steps: int,
    final_input: float,
    initial_input: float = 0.0,
    negligible_dead_time: float = 0.02,
) -> FOPDT:
    """Fit a FOPDT model with the two-point (35.3 % / 85.3 %) method.

    Args:
        negligible_dead_time: Dead times below this fraction of the identified
            time constant are reported as zero.

    Raises:
        IdentificationError: if the input does not change, the plant does not
            respond, or the response is oscillatory (in which case the fit is
            not merely inaccurate but meaningless — use :func:`pole_placement`).
    """
    delta_u = final_input - initial_input
    if delta_u == 0:
        raise IdentificationError("No step change in input.")

    time, output = system.step_response(
        steps=steps, final_input=final_input, initial_input=initial_input
    )
    time_np = time.numpy()
    y = output.numpy()

    y0, y_inf = float(y[0]), float(y[-1])
    delta_y = y_inf - y0
    if delta_y == 0:
        raise IdentificationError("System output does not respond to the input.")

    # A monotone first-order response never exceeds its own final value. A peak
    # meaningfully above it means the plant is oscillatory and the two-point
    # method does not apply.
    peak = float(y.max()) if delta_y > 0 else float(y.min())
    if abs(peak - y0) > 1.05 * abs(delta_y):
        raise IdentificationError(
            "Step response is oscillatory; the two-point FOPDT fit is invalid "
            "for this plant. Use relay autotuning instead."
        )

    # Two-point (Smith) identification at 35.3 % and 85.3 % of the final value.
    #
    # The coefficients belong to this pair of levels and cannot be mixed with
    # another variant's. The earlier code read off 35 % / 85 % but applied
    # T = 1.5*(t85 - t35), which belongs to the 28.3 % / 63.2 % variant: on a
    # pure first-order plant with tau = 100 s that returns T = 222 s and a
    # *negative* dead time, and every gain derived from it inherits the error.
    def time_at(fraction: float) -> float:
        level = y0 + fraction * delta_y
        reached = (y >= level) if delta_y > 0 else (y <= level)
        index = reached.argmax() if reached.any() else None
        if index is None:
            raise IdentificationError(
                f"Output never reaches {fraction:.0%} of its final value."
            )
        return float(time_np[index])

    t_35, t_85 = time_at(0.353), time_at(0.853)

    T = 0.67 * (t_85 - t_35)
    L = max(0.0, 1.3 * t_35 - 0.29 * t_85)

    # A dead time far below the time constant is sampling noise, not transport
    # delay, and the rules that divide by it are extremely sensitive there: an
    # L of 1 s against T = 100 s pushed IMC to Kp = 1072 instead of ~105.
    if L < negligible_dead_time * T:
        L = 0.0

    K = delta_y / delta_u
    return FOPDT(K=K, L=L, T=T)


def relay_autotune(
    system: BaseSystem,
    relay_amplitude: float,
    steps: int,
    setpoint: float = 0.0,
    hysteresis: float = 0.0,
) -> tuple[float, float]:
    """Åström–Hägglund relay experiment. Returns ``(ultimate_gain, ultimate_period)``.

    Drives the plant with a relay around ``setpoint``; the closed loop settles
    into a limit cycle at the plant's ultimate frequency. From the amplitude
    ``a`` of that cycle the describing-function estimate of the ultimate gain is
    ``Ku = 4d / (π·a)``.
    """
    with torch.no_grad():
        system.reset()
        outputs, times, switch_times = [], [], []
        sign = 1.0
        for step in range(steps):
            y = float(system.X.reshape(-1)[0])
            error = setpoint - y
            new_sign = sign
            if error > hysteresis:
                new_sign = 1.0
            elif error < -hysteresis:
                new_sign = -1.0
            if new_sign != sign:
                switch_times.append(step * float(system.dt.reshape(-1)[0]))
            sign = new_sign
            system.apply_control(torch.tensor(sign * relay_amplitude))
            outputs.append(float(system.X.reshape(-1)[0]))
            times.append(step * float(system.dt.reshape(-1)[0]))

    if len(switch_times) < 3:
        raise IdentificationError(
            "Relay experiment did not produce a limit cycle; try more steps or "
            "a larger relay amplitude."
        )

    # Use the last few full cycles, after the transient has died out.
    recent = switch_times[-5:]
    half_periods = [b - a for a, b in zip(recent, recent[1:], strict=False)]
    ultimate_period = 2.0 * (sum(half_periods) / len(half_periods))

    settled = outputs[len(outputs) // 2 :]
    amplitude = (max(settled) - min(settled)) / 2.0
    if amplitude <= 0:
        raise IdentificationError("Limit cycle has zero amplitude.")

    ultimate_gain = 4.0 * relay_amplitude / (torch.pi * amplitude)
    return float(ultimate_gain), float(ultimate_period)


# ── tuning rules ─────────────────────────────────────────────────────────
def ziegler_nichols(model: FOPDT) -> Gains:
    """Open-loop Ziegler–Nichols rule.

    Kp = 1.2·T/(K·L), Ti = 2L, Td = 0.5L.

    Note that Ki and Kd are the *parallel-form* gains Kp/Ti and Kp·Td. The
    earlier implementation returned Kd = 0.5·L, i.e. the derivative *time*
    rather than the derivative gain, which is wrong by a factor of Kp.
    """
    if model.L <= 0 or model.K == 0:
        raise IdentificationError(
            "Ziegler–Nichols needs a non-zero dead time and process gain."
        )
    Kp = 1.2 * model.T / (model.K * model.L)
    Ti = 2.0 * model.L
    Td = 0.5 * model.L
    return Kp, Kp / Ti, Kp * Td


def cohen_coon(model: FOPDT) -> Gains:
    """Cohen–Coon rule, tuned for larger L/T ratios than Ziegler–Nichols."""
    if model.L <= 0 or model.K == 0:
        raise IdentificationError(
            "Cohen–Coon needs a non-zero dead time and process gain."
        )
    ratio = model.T / model.L
    Kp = (1.0 / model.K) * ((1.35 * ratio) + 0.27)
    Ti = model.T * ((2.5 * (model.L / model.T) + 0.9) / (1 + 0.6 * (model.L / model.T)))
    Td = 0.37 * model.L * (model.T / (model.T + 0.2 * model.L))
    return Kp, Kp / Ti, Kp * Td


def pid_imc(model: FOPDT, lambda_value: float | None = None) -> Gains:
    """Internal Model Control (SIMC) tuning.

    ``lambda_value`` is the desired closed-loop time constant; it is clamped
    away from zero so the rule cannot return infinite gain on a plant that
    identifies with no dead time.
    """
    if model.K == 0:
        raise IdentificationError("IMC needs a non-zero process gain.")

    if model.L <= 0:
        # First order without dead time: PI is sufficient, D adds only noise.
        lam = max(lambda_value or 0.0, 0.1 * model.T)
        Kp = model.T / (model.K * lam)
        return Kp, Kp / model.T, 0.0

    lam = max(lambda_value or 0.3 * model.L, 0.8 * model.L)
    Kp = (2.0 * model.T + model.L) / (2.0 * model.K * lam)
    Ti = model.T + model.L / 2.0
    Td = (model.T * model.L) / (2.0 * model.T + model.L)
    return Kp, Kp / Ti, Kp * Td


def ziegler_nichols_ultimate(ultimate_gain: float, ultimate_period: float) -> Gains:
    """Closed-loop Ziegler–Nichols rule from a relay experiment."""
    Kp = 0.6 * ultimate_gain
    Ti = ultimate_period / 2.0
    Td = ultimate_period / 8.0
    return Kp, Kp / Ti, Kp * Td


def pole_placement(system: BaseSystem, bandwidth_factor: float = 2.0,
                   damping: float = 0.8) -> Gains:
    """Place the closed-loop poles analytically from the plant's own parameters.

    Neither step-response identification nor a relay experiment is meaningful for
    the trolley: it has relative degree two and no dead time, so its phase never
    reaches −180° and there *is* no finite ultimate gain. A relay run on it locks
    onto a chattering cycle at the sampling limit and reports an ultimate gain
    that is an artefact of the sample rate, not a property of the plant.

    Since both plants here have known structure, the honest baseline is to solve
    for the gains directly.

    Trolley — m·s³ + (c + Kd)·s² + (k + Kp)·s + Ki, matched against
    (s + ω)(s² + 2ζω·s + ω²):

        Kd = m(1 + 2ζ)ω − c,  Kp = m(1 + 2ζ)ω² − k,  Ki = m·ω³

    Thermal — C·s² + (h + Kp)·s + Ki, matched against C(s² + 2ζω·s + ω²):

        Kp = 2ζωC − h,  Ki = C·ω²,  Kd = 0

    Args:
        bandwidth_factor: Closed-loop bandwidth as a multiple of the plant's own
            characteristic frequency. Higher is faster and hits the actuator
            limits sooner.
        damping: Target damping ratio ζ of the dominant pole pair.
    """
    from entities.systems import Thermal, Trolley

    if isinstance(system, Trolley):
        m = float(system.mass)
        k = float(system.spring)
        c = float(system.friction)
        omega_n = (k / m) ** 0.5
        omega = bandwidth_factor * max(omega_n, 1e-6)
        Kd = m * (1.0 + 2.0 * damping) * omega - c
        Kp = m * (1.0 + 2.0 * damping) * omega**2 - k
        Ki = m * omega**3
        return max(Kp, 0.0), max(Ki, 0.0), max(Kd, 0.0)

    if isinstance(system, Thermal):
        C = float(system.thermal_capacity)
        h = float(system.heat_transfer_coefficient)
        omega = bandwidth_factor / float(system.tau)
        Kp = 2.0 * damping * omega * C - h
        Ki = C * omega**2
        return max(Kp, 0.0), max(Ki, 0.0), 0.0

    raise IdentificationError(
        f"No pole-placement design for {type(system).__name__}; its structure "
        "is not known to this module."
    )


# ── entry point ──────────────────────────────────────────────────────────
def tune(
    system: BaseSystem,
    method: TuningMethod,
    steps: int,
    step_input: float,
    relay_amplitude: float | None = None,
    lambda_value: float | None = None,
) -> Gains:
    """Tune a PID for ``system``.

    ``method="auto"`` identifies the plant first and falls back to analytic pole
    placement when the step response turns out to be oscillatory — which is what
    makes the baseline fair across both plants in this project.
    """
    amplitude = relay_amplitude if relay_amplitude is not None else abs(step_input)

    if method == "pole_placement":
        return pole_placement(system)

    if method == "relay":
        Ku, Tu = relay_autotune(system, amplitude, steps)
        return ziegler_nichols_ultimate(Ku, Tu)

    try:
        model = identify_fopdt(system, steps=steps, final_input=step_input)
    except IdentificationError:
        if method != "auto":
            raise
        # The step test could not identify the plant — which for a plant with
        # no dead time also rules out the relay method. Fall back to solving
        # for the gains from the plant's structure.
        return pole_placement(system)

    if method in ("pid_imc", "auto"):
        return pid_imc(model, lambda_value)
    if method == "ziegler_nichols":
        return ziegler_nichols(model)
    if method == "cohen_coon":
        return cohen_coon(model)
    raise ValueError(f"Unknown tuning method: {method!r}")
