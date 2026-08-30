"""How much is there for an adaptive controller to win here at all?

    python -m comparisons.headroom trolley

This is the diagnostic that decides whether a gain scheduler is worth having on
a given problem, and it should be run *before* concluding anything from a
scheduler that fails to beat constant gains.

There are two different things a scheduler could exploit, and they need
measuring separately:

**Between-episode headroom.** One global constant, against the best constant for
each episode *individually* (searched on that episode alone, with full knowledge
of its plant). This is the most any method can win by recognising which plant it
is on. On both linear studies it comes out at roughly zero: the optimal constant
barely moves across the randomisation range, so no amount of training can turn a
profit there.

**Within-episode headroom.** The best constant for an episode, against the best
*schedule* — a lookup table of gains keyed on the operating point — for that same
episode. This is what gain scheduling proper is for, and it is the number that
matters on a nonlinear plant, where the right gains depend on where the plant is
sitting rather than on which plant it is.

Both are computed from the plant and the objective alone. Neither involves the
network, which is the point: run this *before* concluding anything from a
scheduler that fails to beat constant gains, because a small headroom means the
result describes the problem and not the method.

Why the between-episode gap is near zero here: within one episode these plants
are linear and time-invariant unless the nonlinear terms are switched on, and
for a fixed linear plant the optimal fixed-structure controller is constant.
Time-varying gains have nothing to exploit except actuator saturation and the
heater's inability to cool.
"""

import argparse
import json
import os

import numpy as np

from comparisons import baselines
from comparisons.compare import Arm, episode_cost, run_arm
from config import available_studies, cnfg, load_config
from learning.scenarios import make_episode
from learning.utils import N_FEATURES
from models.pid_lstm import LSTMAdaptivePID
from utils import save_load
from utils.seeding import DEFAULT_SEED, seed_everything

HEADROOM_SEED_BASE = 700_000


def main(system_name: str, seed: int, runs: int, iterations: int,
         n_bins: int) -> None:
    seed_everything(seed)
    config = load_config(system_name)
    dt = config.learning.dt
    steps = int(config.learning.lstm.train_time / dt)
    ceiling = config.control.gain_ceiling

    rbf_model = save_load.load_rbf_model(f"sys_rbf_{system_name}.pth")

    # The headroom itself does not involve the network - that is the point of
    # the measurement, and it is why this runs before any training.
    lstm_model = None
    checkpoint = os.path.join(cnfg.WEIGHTS_DIR, f"pid_lstm_{system_name}.pth")
    if os.path.exists(checkpoint):
        lstm_model = LSTMAdaptivePID(
            input_size=N_FEATURES,
            hidden_size=config.learning.lstm.model.hidden_size,
            output_size=3,
            num_layers=config.learning.lstm.model.num_layers,
        )
        save_load.load_model(lstm_model, f"pid_lstm_{system_name}.pth")
    else:
        print(f"No trained scheduler at {checkpoint} — reporting the ceiling only.")

    rng = np.random.default_rng(HEADROOM_SEED_BASE + seed)
    episodes = [make_episode(config.scenario, steps, dt, rng) for _ in range(runs)]

    def cost(gains, episode) -> float:
        arm = Arm(name="probe", gains=gains)
        results = run_arm(arm, config, rbf_model, episode, with_disturbance=True)
        return episode_cost(results, dt)

    # ── 1. one constant for the whole distribution ──────────────────────
    print(f"Searching one global constant over {runs} episodes...")
    global_gains, _ = baselines.optimise_fixed_gains(
        lambda g: baselines.mean_objective(cost(g, e) for e in episodes),
        ceiling=ceiling,
        iterations=iterations,
        seed=seed,
    )
    global_costs = [cost(global_gains, e) for e in episodes]
    print(f"  gains: Kp={global_gains[0]:.3f} Ki={global_gains[1]:.3f} "
          f"Kd={global_gains[2]:.3f}")

    # ── 2. the best constant for each episode on its own ────────────────
    print(f"\nSearching a separate constant for each of {runs} episodes...")
    oracle_costs, oracle_gains = [], []
    for index, episode in enumerate(episodes):
        gains, score = baselines.optimise_fixed_gains(
            lambda g, e=episode: cost(g, e),
            ceiling=ceiling,
            iterations=iterations,
            seed=seed + index,
        )
        oracle_costs.append(score)
        oracle_gains.append(gains)
        print(f"  episode {index + 1:>2}/{runs}  IAE {score:9.3f}  "
              f"(global constant: {global_costs[index]:9.3f})")

    # ── 3. the best lookup-table schedule per episode ───────────────────
    print(f"\nSearching a {n_bins}-bin schedule for each episode "
          f"(seeded with that episode's best constant)...")
    schedule_costs = []
    for index, episode in enumerate(episodes):
        def scheduled_cost(table, e=episode) -> float:
            arm = Arm(name="sched", gains=None, lstm=baselines.ScheduledGains(table))
            return episode_cost(
                run_arm(arm, config, rbf_model, e, with_disturbance=True), dt
            )

        _, score = baselines.optimise_schedule(
            scheduled_cost,
            n_bins=n_bins,
            # More parameters need more evaluations to be given a fair chance.
            iterations=iterations * n_bins,
            seed=seed + index,
            start_from=oracle_gains[index],
            ceiling=ceiling,
        )
        schedule_costs.append(min(score, oracle_costs[index]))
        print(f"  episode {index + 1:>2}/{runs}  IAE {score:9.3f}  "
              f"(best constant for it: {oracle_costs[index]:9.3f})")

    # ── 4. the trained scheduler, if one has been trained yet ───────────
    if lstm_model is None:
        lstm_costs = [float("nan")] * len(episodes)
    else:
        scheduler_arm = Arm(name="lstm", gains=None, lstm=lstm_model)
        lstm_costs = [
            episode_cost(run_arm(scheduler_arm, config, rbf_model, episode, True), dt)
            for episode in episodes
        ]

    # ── report ──────────────────────────────────────────────────────────
    g, o, adaptive = map(np.array, (global_costs, oracle_costs, lstm_costs))
    sched = np.array(schedule_costs)
    headroom = (g.mean() - o.mean()) / g.mean() * 100.0
    within = (o.mean() - sched.mean()) / o.mean() * 100.0
    total_headroom = (g.mean() - sched.mean()) / g.mean() * 100.0
    captured = (g.mean() - adaptive.mean()) / g.mean() * 100.0

    print(f"\n{'=' * 66}\nAdaptation headroom — {system_name}\n{'=' * 66}")
    print(f"  one global constant            IAE {g.mean():10.3f}")
    print(f"  best constant per episode      IAE {o.mean():10.3f}")
    print(f"  best {n_bins}-bin schedule per ep.   IAE {sched.mean():10.3f}")
    if lstm_model is not None:
        print(f"  LSTM scheduler                 IAE {adaptive.mean():10.3f}")
    print()
    print(f"  between-episode headroom       {headroom:9.1f} %   "
          f"(knowing which plant you are on)")
    print(f"  within-episode headroom        {within:9.1f} %   "
          f"(scheduling on the operating point)")
    print(f"  total available                {total_headroom:9.1f} %")
    if lstm_model is not None:
        print(f"  captured by the scheduler      {captured:9.1f} %")
        if total_headroom > 1.0:
            print(f"  fraction of it captured        "
                  f"{captured / total_headroom * 100:9.1f} %")

    spread = np.array(oracle_gains)
    print("\n  Spread of the per-episode optimal gains "
          "(how much the best constant actually moves):")
    for index, name in enumerate(("Kp", "Ki", "Kd")):
        column = spread[:, index]
        rel = column.std() / column.mean() if column.mean() else float("nan")
        print(f"    {name}  min {column.min():8.3f}  max {column.max():8.3f}  "
              f"mean {column.mean():8.3f}  cv {rel:5.2f}")

    print("\n  Reading: headroom near zero on both lines means no adaptive "
          "controller\n  can beat a constant here, however well it is trained.")

    path = f"{cnfg.METRICS_DIR}/headroom_{system_name}.json"
    with open(path, "w") as handle:
        json.dump(
            {
                "system": system_name,
                "seed": seed,
                "runs": runs,
                "global_gains": list(global_gains),
                "global_iae_mean": float(g.mean()),
                "per_episode_iae_mean": float(o.mean()),
                "per_episode_schedule_iae_mean": float(sched.mean()),
                "within_episode_headroom_percent": float(within),
                "total_headroom_percent": float(total_headroom),
                "n_bins": n_bins,
                "lstm_iae_mean": float(adaptive.mean()),
                "headroom_percent": float(headroom),
                "captured_percent": float(captured),
                "per_episode_gains": [list(x) for x in oracle_gains],
            },
            handle,
            indent=2,
        )
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=available_studies())
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--bins", type=int, default=4,
                        help="Operating-point bins in the schedule oracle.")
    args = parser.parse_args()
    main(args.system, args.seed, args.runs, args.iterations, args.bins)
