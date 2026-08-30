"""Compare the LSTM gain scheduler against fixed-gain baselines.

    python -m comparisons.compare trolley
    python -m comparisons.compare thermal --runs 40

Protocol, which is most of what makes the numbers meaningful:

* Every controller is run on the **same** held-out episodes — same reference
  staircase, same disturbances, same plant parameters. The previous version drew
  a fresh random setpoint per controller and, for the third arm, skipped the
  reset entirely, so that run started from wherever the previous one had left
  the trolley and the controller kept the gains the LSTM had last set.
* The plant and controller are reset before every single run.
* Because the episodes are shared, comparisons are **paired**: alongside the
  means we report how often each controller wins episode-by-episode, which is
  robust to one outlier run dominating a mean.
* Two protocols are scored separately (with and without load disturbance),
  because shape metrics like settling time are undefined under a disturbance
  that never stops moving.
"""

import argparse
import json
from dataclasses import dataclass

import numpy as np
import torch

from classes.simulation import SimulationConfig, SimulationResults
from comparisons import baselines
from config import available_studies, cnfg, load_config
from config.models import ConfigPack
from entities.pid import PID
from learning.scenarios import Episode, build_system, make_episode
from learning.utils import N_FEATURES, extract_lstm_input
from learning.utils.extract_rbf_input import EXTRACTORS
from models.pid_lstm import LSTMAdaptivePID
from utils import save_load
from utils.metrics import LABELS, StepMetrics, aggregate, compute_metrics
from utils.plot import plot_comparison_episode, plot_metric_distributions
from utils.run import run_episode
from utils.seeding import DEFAULT_SEED, seed_everything

COMPARE_SEED_BASE = 500_000  # disjoint from training and from LSTM validation
PRIMARY_METRICS = ("iae", "ise", "itae", "overshoot", "settling_time", "control_effort")


@dataclass
class Arm:
    """One controller under test."""

    name: str
    gains: tuple[float, float, float] | None  # None = gains come from the LSTM
    lstm: torch.nn.Module | None = None
    per_episode_gains: bool = False  # recompute gains from each episode's plant


def build_simulation_config(
    config: ConfigPack, episode: Episode, with_disturbance: bool
) -> SimulationConfig:
    lstm = config.learning.lstm
    return SimulationConfig(
        setpoints=episode.setpoints,
        disturbances=episode.disturbances if with_disturbance else None,
        dt=torch.tensor(config.learning.dt),
        sequence_length=lstm.sequence_length,
        tbptt_window=lstm.tbptt_window,
        warm_up_steps=lstm.warm_up_steps,
        pid_gain_factor=config.control.gain_ceiling,
        error_scale=config.control.error_scale,
        operating_range=config.scenario.setpoint.as_tuple(),
    )


def run_arm(
    arm: Arm,
    config: ConfigPack,
    rbf_model: torch.nn.Module,
    episode: Episode,
    with_disturbance: bool,
) -> SimulationResults:
    """Run one controller on one episode, from a guaranteed-clean state."""
    system = build_system(config, episode.plant_parameters)
    system.reset()

    if arm.per_episode_gains:
        gains = baselines.per_episode_pole_placement(system).gains
        system.reset()  # the tuner runs the plant; undo it
    else:
        gains = arm.gains or config.control.initial_gains

    pid = PID(*baselines.as_tensor_gains(gains))
    pid.set_limits(
        torch.tensor(config.control.output_max),
        torch.tensor(config.control.output_min),
    )
    pid.reset()

    return run_episode(
        system=system,
        pid=pid,
        simulation_config=build_simulation_config(config, episode, with_disturbance),
        extract_rbf_input=EXTRACTORS[config.plant],
        extract_lstm_input=extract_lstm_input,
        rbf_model=rbf_model,
        lstm_model=arm.lstm,
        session="validation" if arm.lstm is not None else "static",
    ).results


def score_final_step(results: SimulationResults, dt: float) -> StepMetrics:
    """Score the last reference segment, whose start is a known step."""
    setpoints = results.as_floats("setpoints")
    positions = results.as_floats("positions")
    controls = results.as_floats("control_outputs")

    final = setpoints[-1]
    start = len(setpoints) - 1
    while start > 0 and setpoints[start - 1] == final:
        start -= 1
    start = min(start, len(setpoints) - 10)
    start = max(start, 0)

    return compute_metrics(
        positions[start:],
        setpoint=final,
        dt=dt,
        control_outputs=controls[start:],
        initial_value=positions[start],
    )


def episode_cost(results: SimulationResults, dt: float) -> float:
    """Whole-episode IAE — the objective the fixed-gain search minimises."""
    errors = np.abs(np.array(results.as_floats("error_history")))
    if not np.isfinite(errors).all():
        return float("inf")
    return float(np.trapezoid(errors, dx=dt))


def win_rate(a: list[StepMetrics], b: list[StepMetrics], field: str) -> float:
    """Fraction of paired episodes where ``a`` beats ``b`` on ``field``.

    Lower is better for every metric reported here. Episodes where either side
    produced no number (a response that never settled, say) are excluded rather
    than scored as a win for whoever happened to produce a value.
    """
    wins = comparable = 0
    for left, right in zip(a, b, strict=True):
        x, y = getattr(left, field), getattr(right, field)
        if np.isnan(x) or np.isnan(y):
            continue
        comparable += 1
        wins += x < y
    return wins / comparable if comparable else float("nan")


def report(protocol: str, per_arm: dict[str, list[StepMetrics]], reference: str) -> dict:
    """Print and return the comparison table for one protocol."""
    summaries = {name: aggregate(runs) for name, runs in per_arm.items()}
    names = list(per_arm)

    print(f"\n{'=' * 78}")
    print(f"Protocol: {protocol}")
    print("=" * 78)
    header = f"{'metric':<26}" + "".join(f"{n[:16]:>17}" for n in names)  # noqa: E501
    print(header)
    print("-" * len(header))

    for key in PRIMARY_METRICS:
        row = f"{LABELS[key][:25]:<26}"
        for name in names:
            value = summaries[name][key]["mean"]
            row += f"{value:>17.3f}" if not np.isnan(value) else f"{'n/a':>17}"
        print(row)

    valid = f"{'(settling time scored in)':<26}"
    for name in names:
        stats = summaries[name]["settling_time"]
        valid += f"{stats['n_valid']}/{stats['n_total']}".rjust(17)
    print(valid)

    print(f"\nPaired win rate vs {reference} (fraction of episodes where the")
    print("LSTM scheduler is strictly better on that metric):")
    lstm_name = next((n for n in names if "LSTM" in n), None)
    if lstm_name:
        for key in PRIMARY_METRICS:
            rate = win_rate(per_arm[lstm_name], per_arm[reference], key)
            bar = "n/a" if np.isnan(rate) else f"{rate:5.0%}"
            print(f"  {LABELS[key][:34]:<36}{bar}")

    return summaries


def main(system_name: str, seed: int, runs: int, show: bool) -> None:
    seed_everything(seed)
    config = load_config(system_name)
    dt = config.learning.dt
    steps = int(config.learning.lstm.train_time / dt)

    rbf_model = save_load.load_rbf_model(f"sys_rbf_{system_name}.pth")
    lstm_model = LSTMAdaptivePID(
        input_size=N_FEATURES,
        hidden_size=config.learning.lstm.model.hidden_size,
        output_size=3,
        num_layers=config.learning.lstm.model.num_layers,
    )
    save_load.load_model(lstm_model, f"pid_lstm_{system_name}.pth")

    # Episodes for tuning the fixed-gain baseline: separate from the ones it is
    # then scored on, so the search cannot overfit the evaluation set.
    tune_rng = np.random.default_rng(COMPARE_SEED_BASE + seed)
    tune_episodes = [
        make_episode(config.scenario, steps, dt, tune_rng) for _ in range(8)
    ]
    eval_rng = np.random.default_rng(COMPARE_SEED_BASE + seed + 7777)
    eval_episodes = [
        make_episode(config.scenario, steps, dt, eval_rng) for _ in range(runs)
    ]

    # ── baselines ───────────────────────────────────────────────────────
    nominal = build_system(config)
    classical = baselines.classical(nominal, config)
    print(
        f"Classical tuning ({classical.name}): "
        f"Kp={classical.gains[0]:.3f} Ki={classical.gains[1]:.3f} "
        f"Kd={classical.gains[2]:.3f}"
    )

    print(
        f"\nSearching for the best constant gains over "
        f"{len(tune_episodes)} episodes..."
    )

    def objective(gains):
        arm = Arm(name="search", gains=gains)
        return baselines.mean_objective(
            episode_cost(
                run_arm(arm, config, rbf_model, episode, True), dt
            )
            for episode in tune_episodes
        )

    best_gains, best_score = baselines.optimise_fixed_gains(
        objective, ceiling=config.control.gain_ceiling, iterations=160, seed=seed
    )
    print(f"  best fixed gains: Kp={best_gains[0]:.3f} Ki={best_gains[1]:.3f} "
          f"Kd={best_gains[2]:.3f}  (mean IAE {best_score:.3f})")

    arms = [
        Arm(name="classical PID", gains=classical.gains),
        Arm(name="best fixed PID", gains=best_gains),
        Arm(name="LSTM scheduled", gains=None, lstm=lstm_model),
        Arm(name="pole place/episode", gains=None, per_episode_gains=True),
    ]

    # ── run both protocols ──────────────────────────────────────────────
    summary = {"system": system_name, "seed": seed, "runs": runs,
               "classical_gains": list(classical.gains),
               "best_fixed_gains": list(best_gains),
               "protocols": {}}

    for protocol, with_disturbance in (("tracking", False), ("rejection", True)):
        per_arm: dict[str, list[StepMetrics]] = {}
        example: dict[str, SimulationResults] = {}
        for arm in arms:
            scores = []
            for index, episode in enumerate(eval_episodes):
                results = run_arm(
                    arm, config, rbf_model, episode, with_disturbance
                )
                scores.append(score_final_step(results, dt))
                if index == 0:
                    example[arm.name] = results
            per_arm[arm.name] = scores

        summary["protocols"][protocol] = report(
            protocol, per_arm, reference="best fixed PID"
        )

        plot_comparison_episode(
            example,
            system_name=system_name.capitalize(),
            protocol=protocol,
            show=show,
        )
        plot_metric_distributions(
            {name: [m.iae for m in runs_] for name, runs_ in per_arm.items()},
            metric_label=LABELS["iae"],
            system_name=system_name.capitalize(),
            protocol=protocol,
            show=show,
        )

    path = f"{cnfg.METRICS_DIR}/comparison_{system_name}.json"
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nResults written to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=available_studies())
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(args.system, args.seed, args.runs, args.show)
