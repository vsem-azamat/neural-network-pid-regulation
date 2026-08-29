"""Dataset construction for the RBF plant surrogate.

The surrogate is used to predict the *next* output along a closed-loop
trajectory, so it has to be trained on trajectories. The original sampler drew
each example independently — a random position, a random velocity, and a random
acceleration that ``apply_control`` immediately overwrote from the force, making
that fourth input pure noise with respect to the label. It also sampled a range
(±100 in position and velocity) an order of magnitude wider than the ±20 the
controller ever visits, so most of the model's capacity went to a region it is
never asked about.

Here the plant is driven with band-limited random inputs from varied initial
conditions, and consecutive states are recorded. The inputs match the feature
extractors in :mod:`learning.utils.extract_rbf_input` exactly.
"""

import numpy as np
import torch

from config.models import ConfigPack
from entities.systems import BaseSystem, Thermal, Trolley
from learning.scenarios import build_system


def _excitation(
    steps: int, dt: float, amplitude: float, hold_seconds: float, rng: np.random.Generator
) -> np.ndarray:
    """Piecewise-constant excitation: rich enough to expose the dynamics."""
    hold = max(1, int(round(hold_seconds / dt)))
    n_levels = int(np.ceil(steps / hold))
    levels = rng.uniform(-amplitude, amplitude, size=n_levels)
    return np.repeat(levels, hold)[:steps].astype(np.float32)


def randomise_initial_state(
    system: BaseSystem, config: ConfigPack, rng: np.random.Generator
) -> None:
    """Place the plant somewhere in its operating range before a rollout.

    The range is taken from the scenario's setpoints, widened by 30 % so the
    dataset also covers the overshoot region the controller transits through.
    """
    low, high = config.scenario.setpoint.as_tuple()
    margin = 0.3 * (high - low)
    low, high = low - margin, high + margin

    if isinstance(system, Trolley):
        system.position = torch.tensor(float(rng.uniform(low, high)))
        system.velocity = torch.tensor(float(rng.uniform(-0.5, 0.5) * (high - low)))
        system.acceleration = torch.tensor(0.0)
    elif isinstance(system, Thermal):
        system.temperature = torch.tensor(float(rng.uniform(low, high)))
        system.temp_derivative = torch.tensor(0.0)


def collect_trajectories(
    system_name: str,
    config: ConfigPack,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(X, y)`` where each row is one state transition.

    ``X`` columns match the plant's feature extractor; ``y`` is the plant output
    one step later.
    """
    rbf = config.learning.rbf
    dt = config.learning.dt
    control = config.control

    features: list[list[float]] = []
    targets: list[float] = []

    for _ in range(rbf.num_trajectories):
        overrides = {
            name: float(rng.uniform(*bounds.as_tuple()))
            for name, bounds in config.scenario.randomize_plant.items()
        }
        system = build_system(system_name, config, overrides)
        system.reset()
        # Start each trajectory somewhere in the operating range rather than
        # always from equilibrium. Without this the dataset only covers the part
        # of the state space a short rollout from rest can reach, and the model
        # extrapolates — badly — everywhere the controller actually drives it.
        randomise_initial_state(system, config, rng)

        # Excite over the range the controller actually commands, so the model
        # is accurate where it will be queried.
        amplitude = max(abs(control.output_min), abs(control.output_max))
        excitation = _excitation(
            rbf.trajectory_steps, dt, amplitude, hold_seconds=10 * dt, rng=rng
        )
        # A heater cannot draw heat out; keep the excitation inside the actuator's
        # real range rather than teaching the model about impossible inputs.
        excitation = np.clip(excitation, control.output_min, control.output_max)

        with torch.no_grad():
            for u in excitation:
                control_tensor = torch.tensor(float(u))
                row = _features(system, control_tensor)
                system.apply_control(control_tensor)
                features.append(row)
                targets.append(float(system.X.reshape(-1)[0]))

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    return X, y


def _features(system: BaseSystem, control: torch.Tensor) -> list[float]:
    if isinstance(system, Trolley):
        return [
            float(system.X),
            float(system.dXdT),
            float(system.d2XdT2),
            float(control),
        ]
    if isinstance(system, Thermal):
        return [float(system.X), float(system.dXdT), float(control)]
    raise TypeError(f"No RBF feature layout for {type(system).__name__}")


def holdout_excitation(
    config: ConfigPack, steps: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """A held-out excitation trajectory for evaluation. Returns ``(time, control)``."""
    rng = np.random.default_rng(seed)
    dt = config.learning.dt
    control = config.control
    amplitude = max(abs(control.output_min), abs(control.output_max))
    excitation = _excitation(steps, dt, amplitude, hold_seconds=10 * dt, rng=rng)
    excitation = np.clip(excitation, control.output_min, control.output_max)
    return np.arange(steps) * dt, excitation


def rollout_comparison(
    model: torch.nn.Module,
    system: BaseSystem,
    controls,
) -> tuple[list[float], list[float]]:
    """One-step-ahead predictions against the true plant along one trajectory.

    Both sequences start from the same state and see the same inputs, so the
    error is attributable to the model alone. Evaluated on a *held-out*
    excitation drawn from the same distribution as training — the original check
    drove the thermal model from 25 (Celsius, while the plant runs in kelvin)
    with inputs spanning the lowest tenth of the training range, so its reported
    error described a regime the controller never enters.
    """
    predicted: list[float] = []
    actual: list[float] = []

    with torch.no_grad():
        for u in controls:
            control = torch.as_tensor(u, dtype=torch.float32).reshape(())
            row = torch.tensor([_features(system, control)], dtype=torch.float32)
            predicted.append(float(model(row).reshape(-1)[0]))
            actual.append(float(system.apply_control(control).reshape(-1)[0]))

    return predicted, actual
