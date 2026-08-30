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


class ScheduledGains(torch.nn.Module):
    """A classical gain schedule: a lookup table keyed on the operating point.

    This is the baseline a *learned* scheduler actually has to beat. Comparing
    against constant gains asks "is scheduling worth anything here?"; comparing
    against this asks "is a neural network worth anything over the table that
    industry has used for gain scheduling for fifty years?" — which is the
    question the project is really about.

    It deliberately presents the same interface as :class:`LSTMAdaptivePID`
    (normalised gains in (0, 1) plus a hidden state), so the simulation loop
    cannot tell them apart and the two are scored through identical code.
    """

    def __init__(self, table: np.ndarray) -> None:
        """
        Args:
            table: ``(n_bins, 3)`` of gain *fractions* in (0, 1). Bin ``i``
                covers an equal slice of the normalised operating point, which
                the loop supplies as feature 6 (the commanded operating point)
                on the range [-1, 1].
        """
        super().__init__()
        self.register_buffer("table", torch.as_tensor(table, dtype=torch.float32))

    @property
    def n_bins(self) -> int:
        return int(self.table.shape[0])

    def forward(self, x: torch.Tensor, hidden=None):
        # Feature 6 of the most recent sample is the commanded operating point,
        # normalised to [-1, 1] by the study's own range.
        operating_point = x[:, -1, 6].clamp(-1.0, 1.0)
        index = ((operating_point + 1.0) / 2.0 * self.n_bins).long()
        index = index.clamp(0, self.n_bins - 1)
        return self.table[index], hidden


def optimise_schedule(
    objective: Callable[[np.ndarray], float],
    n_bins: int = 4,
    iterations: int = 300,
    refinements: int = 3,
    seed: int = 0,
    start_from: Gains | None = None,
    ceiling: Gains | None = None,
) -> tuple[np.ndarray, float]:
    """Search for the best lookup-table schedule, in gain *fractions*.

    Same derivative-free search as :func:`optimise_fixed_gains`, over
    ``n_bins * 3`` parameters instead of 3.

    ``start_from`` seeds the search with a constant schedule — every bin set to
    the same gains. The schedule family *contains* the constant family (all bins
    equal), so a correct measurement of what scheduling buys must start there
    and can then only improve. Starting from the middle of the box instead makes
    the answer depend on the search budget rather than on the problem: with the
    same number of evaluations spread over four times as many parameters, the
    schedule search lands worse than the constant and reports a *negative*
    benefit for a strictly larger family, which is impossible.
    """
    rng = np.random.default_rng(seed)
    shape = (n_bins, 3)

    if start_from is not None and ceiling is not None:
        fractions = np.clip(
            np.array(start_from, dtype=float) / np.array(ceiling, dtype=float),
            1e-3,
            1 - 1e-3,
        )
        best = np.tile(fractions, (n_bins, 1))
    else:
        best = np.full(shape, 0.5)
    best_score = objective(best)

    # Start with a tighter box than the constant search uses: the seed is
    # already a good point, so the budget is better spent refining around it
    # than re-exploring the whole space in 3*n_bins dimensions.
    span = np.full(shape, 0.6)
    centre = best.copy()
    per_round = max(1, iterations // (refinements + 1))

    for _ in range(refinements + 1):
        for _ in range(per_round):
            candidate = np.clip(
                rng.uniform(centre - span / 2, centre + span / 2), 1e-3, 1 - 1e-3
            )
            score = objective(candidate)
            if score < best_score:
                best_score, best = score, candidate
        centre = best.copy()
        span = span / 2.5

    return best, best_score
