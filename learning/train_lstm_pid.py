"""Train the LSTM gain scheduler on top of the RBF surrogate.

    python -m learning.train_lstm_pid trolley
    python -m learning.train_lstm_pid thermal --epochs 80

Each epoch is a freshly generated episode: a new reference staircase, new load
disturbances and a new draw of the plant's physical parameters. That variation
is the point — with a single constant setpoint and a fixed plant, a constant
gain is optimal and there is nothing for a gain scheduler to learn.

The run also evaluates against a fixed-gain baseline on a held-out set of
episodes, so "did training help?" is answered by numbers rather than by eye.
"""

import argparse
import json
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch import optim

from classes.simulation import SimulationConfig
from config import cnfg, load_config
from config.models import ConfigPack
from entities.pid import PID
from learning.scenarios import Episode, build_system, make_episode
from learning.utils import extract_lstm_input
from learning.utils.extract_rbf_input import EXTRACTORS
from models.pid_lstm import LSTMAdaptivePID
from utils import save_load
from utils.metrics import aggregate, compute_metrics
from utils.plot import plot_training_history
from utils.run import run_episode, surrogate_health, tracking_loss
from utils.seeding import DEFAULT_SEED, seed_everything

TRAIN_SEED_BASE = 10_000
EVAL_SEED_BASE = 90_000  # disjoint from training, so evaluation is held out


@dataclass
class EpochRecord:
    epoch: int
    loss: float
    grad_norm: float
    tracking_iae: float


def build_pid(config: ConfigPack) -> PID:
    control = config.control
    pid = PID(*(torch.tensor(g) for g in control.initial_gains))
    pid.set_limits(
        torch.tensor(control.output_max), torch.tensor(control.output_min)
    )
    return pid


def build_simulation_config(
    config: ConfigPack, episode: Episode
) -> SimulationConfig:
    lstm = config.learning.lstm
    return SimulationConfig(
        setpoints=episode.setpoints,
        disturbances=episode.disturbances,
        dt=torch.tensor(config.learning.dt),
        sequence_length=lstm.sequence_length,
        tbptt_window=lstm.tbptt_window,
        warm_up_steps=lstm.warm_up_steps,
        pid_gain_factor=config.control.gain_ceiling,
        error_scale=config.control.error_scale,
    )


def evaluate(
    system_name: str,
    config: ConfigPack,
    lstm_model: torch.nn.Module | None,
    rbf_model: torch.nn.Module,
    episodes: list[Episode],
    with_disturbance: bool = True,
) -> dict:
    """Score a controller across held-out episodes.

    ``lstm_model=None`` runs the fixed-gain baseline on exactly the same
    episodes, which is the only way the two numbers are comparable.

    ``with_disturbance`` selects the protocol, and the two answer different
    questions. Under a load disturbance that keeps changing, no controller ever
    enters a steady band, so settling time and overshoot are undefined for
    everyone and only the integral errors mean anything. Disturbance-free runs
    are where the step-response shape metrics are readable. Reporting one number
    for both protocols at once is how a comparison ends up with a column of NaN.
    """
    extract_rbf = EXTRACTORS[system_name]
    per_episode = []

    for episode in episodes:
        system = build_system(system_name, config, episode.plant_parameters)
        pid = build_pid(config)
        system.reset()
        pid.reset()

        simulation_config = build_simulation_config(config, episode)
        if not with_disturbance:
            simulation_config.disturbances = None
        results = run_episode(
            system=system,
            pid=pid,
            simulation_config=simulation_config,
            extract_rbf_input=extract_rbf,
            extract_lstm_input=extract_lstm_input,
            rbf_model=rbf_model,
            lstm_model=lstm_model,
            session="validation" if lstm_model is not None else "static",
        ).results

        # Score the final setpoint segment: a clean step whose start we know.
        setpoints = results.as_floats("setpoints")
        final = setpoints[-1]
        start = len(setpoints) - 1
        while start > 0 and setpoints[start - 1] == final:
            start -= 1
        if len(setpoints) - start < 10:  # too short to score
            start = max(0, len(setpoints) - 10)

        per_episode.append(
            compute_metrics(
                results.as_floats("positions")[start:],
                setpoint=final,
                dt=float(config.learning.dt),
                control_outputs=results.as_floats("control_outputs")[start:],
                initial_value=results.as_floats("positions")[start],
            )
        )

    return aggregate(per_episode)


KEY_METRICS = (
    "iae",
    "ise",
    "itae",
    "overshoot",
    "rise_time",
    "settling_time",
    "steady_state_error",
    "control_effort",
)


def report_table(protocol: str, scores: dict[str, dict]) -> None:
    """Print one protocol's comparison, with the count of runs behind each mean.

    ``n`` matters: a metric that only converged in 2 of 12 runs has a mean, but
    not one worth quoting, and printing it without the count hides that.
    """
    fixed, adaptive = scores["fixed_gain"], scores["lstm_scheduled"]
    lower_is_better = {"iae", "ise", "itae", "overshoot", "settling_time",
                       "rise_time", "steady_state_error", "control_effort"}

    disturbed = protocol == "rejection"
    print(f"\n  Protocol: {protocol} "
          f"({'with' if disturbed else 'without'} load disturbance)")
    print(f"  {'metric':<22}{'fixed-gain':>14}{'LSTM':>14}{'change':>11}{'n':>7}")
    print("  " + "-" * 68)
    for key in KEY_METRICS:
        f, a = fixed[key]["mean"], adaptive[key]["mean"]
        n = f"{adaptive[key]['n_valid']}/{adaptive[key]['n_total']}"
        if np.isnan(f) or np.isnan(a) or f == 0:
            delta = "  n/a"
        else:
            change = (a - f) / abs(f) * 100.0
            better = (change < 0) == (key in lower_is_better)
            delta = f"{change:+7.1f}% {'+' if better else '-'}"
        print(f"  {key:<22}{f:>14.4f}{a:>14.4f}{delta:>11}{n:>7}")


def main(system_name: str, seed: int, epochs: int | None, show: bool) -> None:
    seed_everything(seed)
    config = load_config(system_name)
    lstm_config = config.learning.lstm
    num_epochs = epochs if epochs is not None else lstm_config.num_epochs
    steps = int(lstm_config.train_time / config.learning.dt)
    extract_rbf = EXTRACTORS[system_name]

    rbf_model = save_load.load_rbf_model(f"sys_rbf_{system_name}.pth")
    for parameter in rbf_model.parameters():
        parameter.requires_grad_(False)  # the surrogate is fixed during control training

    # Warm-start at the classical gains, so training is measured by whether it
    # improves on a competent controller rather than on a random one.
    ceiling = config.control.gain_ceiling
    initial_fraction = tuple(
        g / c
        for g, c in zip(config.control.initial_gains, ceiling)
    )
    lstm_model = LSTMAdaptivePID(
        input_size=5,
        hidden_size=lstm_config.model.hidden_size,
        output_size=3,
        num_layers=lstm_config.model.num_layers,
        dropout=lstm_config.model.dropout,
        initial_gain_fraction=initial_fraction,
    )
    print(f"  warm start at Kp/Ki/Kd = {config.control.initial_gains} "
          f"(fractions {tuple(round(f, 3) for f in initial_fraction)})")

    optimizer_cls = optim.Adam if lstm_config.optimizer.name == "adam" else optim.SGD
    kwargs = (
        {}
        if lstm_config.optimizer.name == "adam"
        else {"momentum": lstm_config.optimizer.momentum}
    )
    optimizer = optimizer_cls(
        lstm_model.parameters(), lr=lstm_config.optimizer.lr, **kwargs
    )
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=lstm_config.scheduler.gamma
    )

    # Held-out evaluation episodes, fixed once so every checkpoint is scored on
    # exactly the same problems.
    eval_rng = np.random.default_rng(EVAL_SEED_BASE + seed)
    eval_episodes = [
        make_episode(config.scenario, steps, config.learning.dt, eval_rng)
        for _ in range(12)
    ]

    print(f"Training on {system_name}: {num_epochs} episodes x {steps} steps")
    print(f"  loss target: {lstm_config.loss_target}")
    history: list[EpochRecord] = []

    for epoch in range(num_epochs):
        rng = np.random.default_rng(TRAIN_SEED_BASE + seed + epoch)
        episode = make_episode(config.scenario, steps, config.learning.dt, rng)

        system = build_system(system_name, config, episode.plant_parameters)
        pid = build_pid(config)
        system.reset()
        pid.reset()

        report = run_episode(
            system=system,
            pid=pid,
            simulation_config=build_simulation_config(config, episode),
            extract_rbf_input=extract_rbf,
            extract_lstm_input=extract_lstm_input,
            rbf_model=rbf_model,
            lstm_model=lstm_model,
            session="train",
            optimizer=optimizer,
            grad_clip=lstm_config.grad_clip,
            loss_function=partial(
                tracking_loss,
                target=lstm_config.loss_target,
                overshoot_weight=lstm_config.overshoot_weight,
                effort_weight=lstm_config.effort_weight,
            ),
        )
        scheduler.step()

        errors = np.abs(np.array(report.results.as_floats("error_history")))
        record = EpochRecord(
            epoch=epoch + 1,
            loss=report.mean_loss,
            grad_norm=report.mean_grad_norm,
            tracking_iae=float(np.trapezoid(errors, dx=config.learning.dt)),
        )
        history.append(record)

        if (epoch + 1) % max(1, num_epochs // 12) == 0 or epoch == 0:
            print(
                f"  epoch {record.epoch:>3}/{num_epochs}  "
                f"loss {record.loss:.5f}  |grad| {record.grad_norm:.3e}  "
                f"IAE {record.tracking_iae:9.2f}"
            )

    save_load.save_model(lstm_model, f"pid_lstm_{system_name}.pth")

    # ── held-out evaluation ─────────────────────────────────────────────
    print("\nEvaluating on held-out episodes...")
    lstm_model.eval()
    protocols = {}
    for name, disturbed in (("tracking", False), ("rejection", True)):
        protocols[name] = {
            "fixed_gain": evaluate(
                system_name, config, None, rbf_model, eval_episodes, disturbed
            ),
            "lstm_scheduled": evaluate(
                system_name, config, lstm_model, rbf_model, eval_episodes, disturbed
            ),
        }
        report_table(name, protocols[name])

    summary = {
        "system": system_name,
        "seed": seed,
        "epochs": num_epochs,
        "loss_target": lstm_config.loss_target,
        "surrogate": surrogate_health(
            run_episode(
                system=build_system(system_name, config, eval_episodes[0].plant_parameters),
                pid=build_pid(config),
                simulation_config=build_simulation_config(config, eval_episodes[0]),
                extract_rbf_input=extract_rbf,
                extract_lstm_input=extract_lstm_input,
                rbf_model=rbf_model,
                lstm_model=lstm_model,
                session="validation",
            ).results
        ),
        "protocols": protocols,
        "history": [vars(r) for r in history],
    }
    path = f"{cnfg.METRICS_DIR}/lstm_{system_name}.json"
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nMetrics written to {path}")

    plot_training_history(
        [vars(r) for r in history], system_name=system_name.capitalize(), show=show
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=["trolley", "thermal"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(args.system, args.seed, args.epochs, args.show)
