"""Step-response metrics for comparing controllers.

Every metric here is defined relative to the *travel* from the initial value to
the setpoint, so it behaves identically for positive and negative steps. The
earlier implementations assumed a positive setpoint and a rise from zero:

* overshoot used ``max(y)``, which reports 100 % for any negative setpoint;
* rise time looked for ``y >= 0.9·|setpoint|``, never reached on a negative
  step, so it returned the full simulation length;
* settling time returned the *first* entry into the tolerance band rather than
  the last exit from it, which certifies a permanently oscillating response as
  settled at the moment it first crosses the setpoint.

Roughly half of the trolley runs used a negative setpoint, so those three were
not edge cases — they were most of the table.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class StepMetrics:
    """Standard step-response quality measures."""

    mse: float
    settling_time: float
    overshoot: float  # percent of the commanded travel
    rise_time: float
    steady_state_error: float
    iae: float
    ise: float
    itae: float
    control_effort: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


LABELS = {
    "mse": "MSE",
    "settling_time": "Settling time t_s (s)",
    "overshoot": "Overshoot M_p (%)",
    "rise_time": "Rise time t_r (s)",
    "steady_state_error": "Steady-state error",
    "iae": "IAE",
    "ise": "ISE",
    "itae": "ITAE",
    "control_effort": "Control effort",
}


def _as_array(values) -> np.ndarray:
    return np.asarray([float(v) for v in values], dtype=float)


def compute_metrics(
    outputs,
    setpoint: float,
    dt: float,
    control_outputs=None,
    initial_value: float | None = None,
    settling_tolerance: float = 0.02,
) -> StepMetrics:
    """Compute step-response metrics for one run.

    Args:
        outputs: Measured plant output over the run.
        setpoint: Commanded value (may be negative, or below ``initial_value``).
        dt: Sample interval, s.
        control_outputs: Optional control signal, used for the effort metric.
        initial_value: Value the step starts from. Defaults to ``outputs[0]``.
        settling_tolerance: Half-width of the settling band, as a fraction of
            the commanded travel.

    Returns:
        A :class:`StepMetrics` record. Times are NaN when the response never
        reaches the corresponding threshold, which keeps "never settled" from
        being silently reported as a finite, flattering number.
    """
    y = _as_array(outputs)
    if y.size == 0:
        raise ValueError("No samples to score.")

    y0 = float(y[0]) if initial_value is None else float(initial_value)
    travel = setpoint - y0
    times = np.arange(y.size) * dt
    error = y - setpoint

    # Direction-normalised progress: 0 at the start, 1 at the setpoint,
    # >1 beyond it. This is what makes every threshold below sign-agnostic.
    if travel == 0:
        progress = np.where(np.isclose(y, setpoint), 1.0, 0.0)
        scale = max(float(np.abs(error).max()), 1e-12)
    else:
        progress = (y - y0) / travel
        scale = abs(travel)

    # Overshoot: furthest excursion past the setpoint, as a percentage.
    overshoot = max(0.0, float(progress.max()) - 1.0) * 100.0

    # Rise time: first crossing of 90 % of the commanded travel.
    rise_idx = np.argmax(progress >= 0.9) if (progress >= 0.9).any() else None
    rise_time = float(times[rise_idx]) if rise_idx is not None else float("nan")

    # Settling time: last time the response *leaves* the tolerance band. A run
    # that re-enters and stays is settled; one that keeps oscillating is not.
    outside = np.abs(error) > settling_tolerance * scale
    if not outside.any():
        settling_time = 0.0
    elif outside[-1]:
        settling_time = float("nan")  # never settles within the horizon
    else:
        settling_time = float(times[np.flatnonzero(outside)[-1] + 1])

    # Steady-state error over the final 10 % of the run.
    tail = max(1, y.size // 10)
    steady_state_error = float(np.mean(np.abs(error[-tail:])))

    effort = 0.0
    if control_outputs is not None:
        u = _as_array(control_outputs)
        if u.size > 1:
            # Total variation: how hard the actuator is being worked.
            effort = float(np.sum(np.abs(np.diff(u))))

    return StepMetrics(
        mse=float(np.mean(error**2)),
        settling_time=settling_time,
        overshoot=overshoot,
        rise_time=rise_time,
        steady_state_error=steady_state_error,
        iae=float(np.trapezoid(np.abs(error), times)),
        ise=float(np.trapezoid(error**2, times)),
        itae=float(np.trapezoid(times * np.abs(error), times)),
        control_effort=effort,
    )


def aggregate(runs: list[StepMetrics]) -> dict[str, dict[str, float]]:
    """Mean / std / median per metric across runs, ignoring NaNs.

    ``n_valid`` reports how many runs actually produced a number, so a metric
    that failed to converge in most runs cannot hide behind a clean-looking mean.
    """
    if not runs:
        return {}
    summary: dict[str, dict[str, float]] = {}
    for field in StepMetrics.__dataclass_fields__:
        values = np.array([getattr(r, field) for r in runs], dtype=float)
        valid = values[~np.isnan(values)]
        summary[field] = {
            "mean": float(np.mean(valid)) if valid.size else float("nan"),
            "std": float(np.std(valid)) if valid.size else float("nan"),
            "median": float(np.median(valid)) if valid.size else float("nan"),
            "n_valid": int(valid.size),
            "n_total": int(values.size),
        }
    return summary
