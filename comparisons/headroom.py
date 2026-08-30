"""How much is there for an adaptive controller to win here at all?

    python -m comparisons.headroom trolley

This is the diagnostic that decides whether a gain scheduler is worth having on
a given problem, and it should be run *before* concluding anything from a
scheduler that fails to beat constant gains.

It measures three costs on the same held-out episodes:

1. **One global constant** — the single Kp/Ki/Kd that is best on average over the
   whole episode distribution. This is what a well-tuned fixed controller gets.
2. **Per-episode constant** — the best constant gains for each episode
   *individually*, found by searching that episode alone with full knowledge of
   its plant parameters and its reference. Still constant within the episode.
3. **The trained scheduler.**

The gap between (1) and (2) is the **adaptation headroom**: the most that any
method can win purely by recognising which plant it is on. It is an upper bound
on the payoff from between-episode adaptation, and it is computed from the plant
and the objective alone — it does not depend on the network at all.

If that gap is small, a scheduler *cannot* beat constant gains by much no matter
how well it is trained, and a negative result says something about the problem
rather than about the method. Reading it the other way round — "the network
failed" — is the mistake this module exists to prevent.

Why the gap can be small: within one episode these plants are linear and
time-invariant, and for a fixed linear plant with a quadratic cost the optimal
fixed-structure controller is constant. Time-varying gains have nothing to
exploit except the nonlinearities that are actually present — actuator
saturation, and the asymmetry of a heater that cannot cool.
"""

import argparse
import json

import numpy as np

from comparisons import baselines
from comparisons.compare import Arm, episode_cost, run_arm
from config import cnfg, load_config
from learning.scenarios import make_episode
from models.pid_lstm import LSTMAdaptivePID
from utils import save_load
from utils.seeding import DEFAULT_SEED, seed_everything

HEADROOM_SEED_BASE = 700_000


def main(system_name: str, seed: int, runs: int, iterations: int) -> None:
    seed_everything(seed)
    config = load_config(system_name)
    dt = config.learning.dt
    steps = int(config.learning.lstm.train_time / dt)
    ceiling = config.control.gain_ceiling

    rbf_model = save_load.load_rbf_model(f"sys_rbf_{system_name}.pth")
    lstm_model = LSTMAdaptivePID(
        input_size=5,
        hidden_size=config.learning.lstm.model.hidden_size,
        output_size=3,
        num_layers=config.learning.lstm.model.num_layers,
    )
    save_load.load_model(lstm_model, f"pid_lstm_{system_name}.pth")

    rng = np.random.default_rng(HEADROOM_SEED_BASE + seed)
    episodes = [make_episode(config.scenario, steps, dt, rng) for _ in range(runs)]

    def cost(gains, episode) -> float:
        results = run_arm(
            Arm(name="probe", gains=gains),
            system_name, config, rbf_model, episode, with_disturbance=True,
        )
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

    # ── 3. the trained scheduler ────────────────────────────────────────
    lstm_costs = [
        episode_cost(
            run_arm(
                Arm(name="lstm", gains=None, lstm=lstm_model),
                system_name, config, rbf_model, episode, True,
            ),
            dt,
        )
        for episode in episodes
    ]

    # ── report ──────────────────────────────────────────────────────────
    g, o, adaptive = map(np.array, (global_costs, oracle_costs, lstm_costs))
    headroom = (g.mean() - o.mean()) / g.mean() * 100.0
    captured = (g.mean() - adaptive.mean()) / g.mean() * 100.0

    print(f"\n{'=' * 66}\nAdaptation headroom — {system_name}\n{'=' * 66}")
    print(f"  one global constant            IAE {g.mean():10.3f}")
    print(f"  best constant per episode      IAE {o.mean():10.3f}")
    print(f"  LSTM scheduler                 IAE {adaptive.mean():10.3f}")
    print()
    print(f"  headroom for ANY adaptation    {headroom:9.1f} %")
    print(f"  captured by the scheduler      {captured:9.1f} %")
    if headroom > 1e-9:
        print(f"  fraction of headroom captured  {captured / headroom * 100:9.1f} %")

    spread = np.array(oracle_gains)
    print("\n  Spread of the per-episode optimal gains "
          "(how much the best constant actually moves):")
    for index, name in enumerate(("Kp", "Ki", "Kd")):
        column = spread[:, index]
        rel = column.std() / column.mean() if column.mean() else float("nan")
        print(f"    {name}  min {column.min():8.3f}  max {column.max():8.3f}  "
              f"mean {column.mean():8.3f}  cv {rel:5.2f}")

    print("\n  Reading: a headroom near zero means no controller that only "
          "adapts\n  between episodes can beat a constant here, however well "
          "it is trained.")

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
    parser.add_argument("system", choices=["trolley", "thermal"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=120)
    args = parser.parse_args()
    main(args.system, args.seed, args.runs, args.iterations)
