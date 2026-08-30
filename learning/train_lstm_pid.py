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
import copy
import json
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch import optim

from classes.simulation import SimulationConfig
from comparisons.compare import best_fixed_gains
from config import available_studies, cnfg, load_config
from config.models import ConfigPack
from entities.pid import PID
from learning.scenarios import Episode, build_system, make_episode
from learning.utils import N_FEATURES, extract_lstm_input
from learning.utils.extract_rbf_input import EXTRACTORS
from models.pid_lstm import LSTMAdaptivePID
from utils import save_load
from utils.metrics import aggregate, compute_metrics
from utils.plot import plot_training_history
from utils.run import run_episode, surrogate_health, tracking_loss
from utils.seeding import DEFAULT_SEED, seed_everything

TRAIN_SEED_BASE = 10_000
VAL_SEED_BASE = 60_000  # checkpoint selection; disjoint from both of the others
EVAL_SEED_BASE = 90_000  # disjoint from training, so evaluation is held out


@dataclass
class EpochRecord:
    epoch: int
    loss: float
    grad_norm: float
    tracking_iae: float
    validation_iae: float | None = None


def validation_iae(
    config: ConfigPack,
    lstm_model: torch.nn.Module,
    rbf_model: torch.nn.Module,
    episodes: list[Episode],
    gains: tuple[float, float, float],
) -> float:
    """Mean whole-episode IAE across the validation episodes.

    The same objective the fixed-gain search minimises, so "is this
    checkpoint better than the baseline?" is asked in the baseline's own
    terms.
    """
    extract_rbf = EXTRACTORS[config.plant]
    dt = float(config.learning.dt)
    scores = []
    for episode in episodes:
        system = build_system(config, episode.plant_parameters)
        pid = build_pid(config, gains)
        system.reset()
        pid.reset()
        results = run_episode(
            system=system,
            pid=pid,
            simulation_config=build_simulation_config(config, episode),
            extract_rbf_input=extract_rbf,
            extract_lstm_input=extract_lstm_input,
            rbf_model=rbf_model,
            lstm_model=lstm_model,
            session="validation",
        ).results
        errors = np.abs(np.array(results.as_floats("error_history")))
        scores.append(float(np.trapezoid(errors, dx=dt)))
    return float(np.mean(scores))


def build_pid(
    config: ConfigPack, gains: tuple[float, float, float] | None = None
) -> PID:
    control = config.control
    pid = PID(*(torch.tensor(g) for g in (gains or control.initial_gains)))
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
        operating_range=config.scenario.setpoint.as_tuple(),
        control_scale=max(
            abs(config.control.output_min), abs(config.control.output_max)
        ),
    )


def evaluate(
    config: ConfigPack,
    lstm_model: torch.nn.Module | None,
    rbf_model: torch.nn.Module,
    episodes: list[Episode],
    with_disturbance: bool = True,
    gains: tuple[float, float, float] | None = None,
) -> dict:
    """Score a controller across held-out episodes.

    ``lstm_model=None`` runs the fixed-gain baseline on exactly the same
    episodes, which is the only way the two numbers are comparable. ``gains``
    sets that baseline (and the warm-up gains of the scheduler arm); the
    training report passes the residual centre for both, so its table answers
    the only interesting question — does the scheduler beat the constant
    controller it started as?

    ``with_disturbance`` selects the protocol, and the two answer different
    questions. Under a load disturbance that keeps changing, no controller ever
    enters a steady band, so settling time and overshoot are undefined for
    everyone and only the integral errors mean anything. Disturbance-free runs
    are where the step-response shape metrics are readable. Reporting one number
    for both protocols at once is how a comparison ends up with a column of NaN.
    """
    extract_rbf = EXTRACTORS[config.plant]
    per_episode = []

    for episode in episodes:
        system = build_system(config, episode.plant_parameters)
        pid = build_pid(config, gains)
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
    extract_rbf = EXTRACTORS[config.plant]

    rbf_model = save_load.load_rbf_model(f"sys_rbf_{system_name}.pth")
    for parameter in rbf_model.parameters():
        parameter.requires_grad_(False)  # the surrogate is fixed during control training

    # Centre the residual scheduler on the best constant gains for this study.
    # The untrained network then *is* the controller it will be compared
    # against, and training can only be judged by the deviation it learns.
    baseline_gains, baseline_iae = best_fixed_gains(
        config, system_name, rbf_model, steps, seed
    )
    lstm_model = LSTMAdaptivePID(
        input_size=N_FEATURES,
        hidden_size=lstm_config.model.hidden_size,
        output_size=3,
        num_layers=lstm_config.model.num_layers,
        dropout=lstm_config.model.dropout,
        baseline_gains=baseline_gains,
        gain_ceiling=config.control.gain_ceiling,
        residual_range=config.control.residual_range,
    )
    print(f"  residual baseline Kp/Ki/Kd = "
          f"{tuple(round(g, 2) for g in baseline_gains)} "
          f"(search IAE {baseline_iae:.3f}), "
          f"correction band x{config.control.residual_range}")

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

    # Validation episodes for checkpoint selection: disjoint from the training
    # stream and from the final evaluation set, so selecting on them biases
    # neither.
    val_rng = np.random.default_rng(VAL_SEED_BASE + seed)
    val_episodes = [
        make_episode(config.scenario, steps, config.learning.dt, val_rng)
        for _ in range(6)
    ]
    val_interval = 5

    print(f"Training on {system_name}: {num_epochs} episodes x {steps} steps")
    print(f"  loss target: {lstm_config.loss_target}")
    history: list[EpochRecord] = []

    # The untrained network is exactly the baseline controller, and it is the
    # first checkpoint candidate. Training episodes are noisy enough that the
    # final epoch is routinely not the best one — saving it unconditionally
    # let a wandering network ship a controller *worse* than its own starting
    # point, which selection against the baseline makes impossible (up to the
    # validation/test gap).
    lstm_model.eval()
    best_val = validation_iae(
        config, lstm_model, rbf_model, val_episodes, baseline_gains
    )
    lstm_model.train()
    best_state = copy.deepcopy(lstm_model.state_dict())
    best_epoch = 0
    print(f"  validation IAE at the baseline (epoch 0): {best_val:9.2f}")

    for epoch in range(num_epochs):
        rng = np.random.default_rng(TRAIN_SEED_BASE + seed + epoch)
        episode = make_episode(config.scenario, steps, config.learning.dt, rng)

        system = build_system(config, episode.plant_parameters)
        pid = build_pid(config, baseline_gains)
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
                gain_rate_weight=lstm_config.gain_rate_weight,
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

        if (epoch + 1) % val_interval == 0 or epoch + 1 == num_epochs:
            lstm_model.eval()
            record.validation_iae = validation_iae(
                config, lstm_model, rbf_model, val_episodes, baseline_gains
            )
            lstm_model.train()
            if record.validation_iae < best_val:
                best_val = record.validation_iae
                best_state = copy.deepcopy(lstm_model.state_dict())
                best_epoch = epoch + 1

        history.append(record)

        if (epoch + 1) % max(1, num_epochs // 12) == 0 or epoch == 0:
            val_note = (
                f"  val {record.validation_iae:9.2f}"
                if record.validation_iae is not None
                else ""
            )
            print(
                f"  epoch {record.epoch:>3}/{num_epochs}  "
                f"loss {record.loss:.5f}  |grad| {record.grad_norm:.3e}  "
                f"IAE {record.tracking_iae:9.2f}{val_note}"
            )

    lstm_model.load_state_dict(best_state)
    if best_epoch == 0:
        print(f"\n  Selected checkpoint: the baseline (epoch 0), val IAE "
              f"{best_val:.2f} — no epoch improved on it.")
    else:
        print(f"\n  Selected checkpoint: epoch {best_epoch}, "
              f"val IAE {best_val:.2f}")
    save_load.save_model(lstm_model, f"pid_lstm_{system_name}.pth")

    # ── held-out evaluation ─────────────────────────────────────────────
    print("\nEvaluating on held-out episodes...")
    lstm_model.eval()
    protocols = {}
    for name, disturbed in (("tracking", False), ("rejection", True)):
        protocols[name] = {
            "fixed_gain": evaluate(
                config, None, rbf_model, eval_episodes, disturbed,
                gains=baseline_gains,
            ),
            "lstm_scheduled": evaluate(
                config, lstm_model, rbf_model, eval_episodes, disturbed,
                gains=baseline_gains,
            ),
        }
        report_table(name, protocols[name])

    # One representative episode, to record how well the surrogate tracked the
    # plant under the trained controller.
    probe = eval_episodes[0]
    probe_results = run_episode(
        system=build_system(config, probe.plant_parameters),
        pid=build_pid(config, baseline_gains),
        simulation_config=build_simulation_config(config, probe),
        extract_rbf_input=extract_rbf,
        extract_lstm_input=extract_lstm_input,
        rbf_model=rbf_model,
        lstm_model=lstm_model,
        session="validation",
    ).results

    summary = {
        "system": system_name,
        "seed": seed,
        "epochs": num_epochs,
        "loss_target": lstm_config.loss_target,
        "residual_baseline": list(baseline_gains),
        "residual_range": config.control.residual_range,
        "selected_epoch": best_epoch,
        "selected_validation_iae": best_val,
        "surrogate": surrogate_health(probe_results),
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
    parser.add_argument("system", choices=available_studies())
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(args.system, args.seed, args.epochs, args.show)
