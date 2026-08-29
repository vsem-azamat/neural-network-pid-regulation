"""Episode generation: reference trajectories, load disturbances, plant variation.

An adaptive controller can only be shown to adapt if the episode gives it
something to adapt to. The original training loop built its reference as
``[torch.randn(1) * 10] * train_steps``, which repeats a *single object* — one
step to a random constant, then a hundred and fifty steps of holding it, with no
disturbance and a fixed plant. On that episode a well-chosen constant gain is
optimal, so there is nothing for gain scheduling to win and nothing to measure.

Each generator here is deterministic given a seeded RNG, so a run reproduces
exactly.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch

from config.models import ConfigPack, ScenarioConfig
from entities.systems import BaseSystem, Thermal, Trolley


@dataclass(frozen=True)
class Episode:
    """One generated scenario: what to track, and what fights back."""

    setpoints: list[torch.Tensor]
    disturbances: list[torch.Tensor]
    plant_parameters: dict[str, float]

    def __len__(self) -> int:
        return len(self.setpoints)


def piecewise_constant(
    steps: int,
    dt: float,
    value_range: tuple[float, float],
    segment_range: tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """A staircase of random levels held for random durations."""
    values = np.empty(steps, dtype=np.float32)
    filled = 0
    while filled < steps:
        hold = int(round(rng.uniform(*segment_range) / dt))
        hold = max(1, min(hold, steps - filled))
        values[filled : filled + hold] = rng.uniform(*value_range)
        filled += hold
    return values


def make_episode(
    scenario: ScenarioConfig,
    steps: int,
    dt: float,
    rng: np.random.Generator,
    with_disturbance: bool = True,
    with_randomisation: bool = True,
) -> Episode:
    """Build one episode from the scenario configuration."""
    setpoints = piecewise_constant(
        steps, dt, scenario.setpoint.as_tuple(), scenario.segment_time.as_tuple(), rng
    )

    if with_disturbance and scenario.disturbance_scale > 0:
        levels = piecewise_constant(
            steps,
            dt,
            (-scenario.disturbance_scale, scenario.disturbance_scale),
            (scenario.disturbance_hold, scenario.disturbance_hold),
            rng,
        )
    else:
        levels = np.zeros(steps, dtype=np.float32)

    parameters = {}
    if with_randomisation:
        for name, bounds in scenario.randomize_plant.items():
            parameters[name] = float(rng.uniform(*bounds.as_tuple()))

    return Episode(
        setpoints=[torch.tensor(float(v)) for v in setpoints],
        disturbances=[torch.tensor(float(v)) for v in levels],
        plant_parameters=parameters,
    )


def build_system(
    name: str, config: ConfigPack, overrides: dict | None = None
) -> BaseSystem:
    """Instantiate a plant from config, with optional per-episode overrides."""
    parameters = dict(config.system)
    parameters.update(overrides or {})
    dt = torch.tensor(config.learning.dt)

    if name == "trolley":
        return Trolley(
            mass=torch.tensor(parameters["mass"]),
            spring=torch.tensor(parameters["spring"]),
            friction=torch.tensor(parameters["friction"]),
            dt=dt,
        )
    if name == "thermal":
        return Thermal(
            thermal_capacity=torch.tensor(parameters["thermal_capacity"]),
            heat_transfer_coefficient=torch.tensor(
                parameters["heat_transfer_coefficient"]
            ),
            dt=dt,
            initial_temperature=torch.tensor(parameters["initial_temperature"]),
            ambient_temperature=torch.tensor(parameters["ambient_temperature"]),
        )
    raise ValueError(f"Unknown system: {name!r}")


def episode_stream(
    scenario: ScenarioConfig,
    steps: int,
    dt: float,
    count: int,
    seed: int,
    **kwargs,
) -> Iterable[Episode]:
    """Yield ``count`` independent episodes from one seed."""
    rng = np.random.default_rng(seed)
    for _ in range(count):
        yield make_episode(scenario, steps, dt, rng, **kwargs)
