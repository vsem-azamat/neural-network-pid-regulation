"""Controller baselines the LSTM scheduler is measured against.

Three of them, in increasing order of difficulty:

1. **Classical** — gains from an identification-based rule (:mod:`utils.tuning`).
   The traditional comparison, and the weakest: tuning rules are deliberately
   conservative and are applied to the *nominal* plant, not the randomised one.
2. **Best fixed gains** — the single Kp/Ki/Kd triple that minimises the same
   objective the network is trained on, found by search over the same training
   episodes. This is the baseline that actually matters. Beating a tuning rule
   shows only that the rule is conservative; beating the best possible constant
   gains is the only result that supports the project's premise, since that is
   precisely what a gain *scheduler* has to be worth.
3. **Pole placement per episode** — recomputed from each episode's *true* plant
   parameters, which the scheduler has to infer from the loop's behaviour
   instead. This isolates the value of knowing the plant from the value of
   adapting: note that it is not an upper bound, because placing poles at a
   fixed bandwidth is not the same as minimising tracking error, and in practice
   it scores below the searched fixed gains.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch

from config.models import ConfigPack
from entities.systems import BaseSystem
from utils import tuning

Gains = tuple[float, float, float]


@dataclass(frozen=True)
class Baseline:
    """A fixed-gain controller, plus how its gains were arrived at."""

    name: str
    gains: Gains
    description: str


def classical(system: BaseSystem, config: ConfigPack, method: str = "auto") -> Baseline:
    """Tune on the nominal plant, with a step test long enough to be valid.

    The test length has to come from the plant's own time scale. A fixed 600
    samples is only 3τ on the thermal plant: the response has not reached its
    final value, so the identified gain and time constant are both low and a
    spurious 5 s "dead time" appears — enough to survive the negligible-dead-time
    filter and send IMC down its dead-time branch, which returned
    Kp=227, Kd=553 instead of Kp=100, Kd=0. The classical baseline was being
    made to look bad by the measurement, not by the method.
    """
    gains = tuning.tune(
        system,
        method,  # type: ignore[arg-type]
        steps=tuning.step_test_steps(system),
        step_input=config.control.tuning_step_input,
    )
    return Baseline(
        name=f"classical ({method})",
        gains=gains,
        description="Identification-based tuning rule on the nominal plant.",
    )


def optimise_fixed_gains(
    objective: Callable[[Gains], float],
    ceiling: Gains,
    iterations: int = 240,
    refinements: int = 3,
    seed: int = 0,
) -> tuple[Gains, float]:
    """Search for the constant gains that minimise ``objective``.

    Random search inside the gain box, then successive refinement around the
    incumbent. Deliberately derivative-free: the objective is an average over
    whole episodes with saturation and disturbances in it, so it is neither
    smooth nor cheap to differentiate, and a few hundred evaluations is enough
    to find a strong constant baseline.

    Returns:
        The best gains found and their objective value.
    """
    rng = np.random.default_rng(seed)
    ceiling_array = np.array(ceiling, dtype=float)

    best_gains = tuple(ceiling_array * 0.5)
    best_score = objective(best_gains)  # type: ignore[arg-type]

    span = ceiling_array.copy()
    centre = ceiling_array * 0.5
    per_round = max(1, iterations // (refinements + 1))

    for _ in range(refinements + 1):
        for _ in range(per_round):
            candidate = rng.uniform(
                np.maximum(centre - span / 2, 0.0),
                np.minimum(centre + span / 2, ceiling_array),
            )
            score = objective(tuple(candidate))  # type: ignore[arg-type]
            if score < best_score:
                best_score, best_gains = score, tuple(candidate)
        # Contract the search box around the incumbent.
        centre = np.array(best_gains, dtype=float)
        span = span / 2.5

    return best_gains, best_score  # type: ignore[return-value]


def per_episode_pole_placement(system: BaseSystem) -> Baseline:
    """Pole placement from this episode's true plant parameters."""
    return Baseline(
        name="pole placement (per-episode)",
        gains=tuning.pole_placement(system),
        description="Pole placement given the episode's actual plant parameters.",
    )


def mean_objective(scores: Iterable[float]) -> float:
    """Mean over episodes, treating a diverged episode as a large finite penalty.

    A run that blows up produces inf or NaN; letting either through would make
    every candidate compare equal and turn the search into a coin flip.
    """
    values = np.array(list(scores), dtype=float)
    values = np.where(np.isfinite(values), values, 1e9)
    return float(values.mean())


def as_tensor_gains(gains: Gains) -> tuple[torch.Tensor, ...]:
    return tuple(torch.tensor(float(g)) for g in gains)
