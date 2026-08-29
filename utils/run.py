"""Simulation and training loops.

The training loop is the heart of the project, and the part that previously did
nothing: the loss was computed from tensors that had been rebuilt with
``torch.tensor(...)``, which severs the autograd graph, so ``backward()`` left
every LSTM parameter with ``grad=None`` and ``optimizer.step()`` was a no-op for
ten epochs. The chain that has to stay connected is

    LSTM gains -> PID -> plant state -> RBF surrogate -> loss

and :func:`run_episode` now keeps it connected, truncating it deliberately at
fixed windows (truncated backpropagation through time) instead of accidentally
at every step.

``tests/test_gradient_flow.py`` asserts the gradient is non-zero, so this can
never silently regress again.
"""

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import torch
from torch import nn, optim
from torch.optim.optimizer import Optimizer

from classes.simulation import SimulationConfig, SimulationResults
from entities.pid import PID
from entities.systems import BaseSystem
from models.sys_rbf import SystemRBFModel

Session = Literal["train", "validation", "static"]
RbfExtractor = Callable[[BaseSystem, torch.Tensor], torch.Tensor]
LstmExtractor = Callable[[SimulationConfig, SimulationResults], torch.Tensor]
LossFn = Callable[[SimulationResults, SimulationConfig, int, int], torch.Tensor]


@dataclass
class EpisodeReport:
    """What one episode produced, beyond the raw trajectory."""

    results: SimulationResults
    window_losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)

    @property
    def mean_loss(self) -> float:
        return (
            sum(self.window_losses) / len(self.window_losses)
            if self.window_losses
            else float("nan")
        )

    @property
    def mean_grad_norm(self) -> float:
        return (
            sum(self.grad_norms) / len(self.grad_norms)
            if self.grad_norms
            else float("nan")
        )


def tracking_loss(
    results: SimulationResults,
    config: SimulationConfig,
    window_start: int,
    window_end: int,
    overshoot_weight: float = 0.5,
    effort_weight: float = 0.0,
    target: Literal["plant", "surrogate"] = "plant",
) -> torch.Tensor:
    """Tracking loss over one TBPTT window, normalised by the setpoint scale.

    Args:
        target: ``"plant"`` scores the true plant output. The plants here are
            written in torch, so the simulation is itself differentiable and the
            gradient is exact — this is the default because it is the stronger
            and more honest signal. ``"surrogate"`` scores the RBF model's
            prediction instead, which is what one is forced to do when the plant
            is not differentiable; running both is how this project quantifies
            what the learned model costs.

            The surrogate path is only as good as the surrogate: an RBF whose
            basis functions have saturated returns a near-constant prediction,
            and a constant has zero gradient, so training silently stops. See
            :func:`surrogate_health`.

    The error is divided by the reference magnitude so that episodes with large
    setpoints do not dominate the gradient purely by scale — without this the
    thermal plant (setpoints in the hundreds of kelvin) trains on a loss three
    orders of magnitude larger than the trolley's.
    """
    source = results.rbf_predictions if target == "surrogate" else results.positions
    predicted = torch.stack(source[window_start:window_end])
    reference = torch.stack(
        [torch.as_tensor(s).reshape(-1)[0] for s in results.setpoints[window_start:window_end]]
    )

    scale = reference.abs().mean().detach().clamp(min=1.0)
    error = (predicted - reference) / scale

    loss = torch.mean(error**2) + overshoot_weight * torch.mean(torch.relu(error))

    if effort_weight:
        controls = torch.stack(results.control_outputs[window_start:window_end])
        if controls.numel() > 1:
            loss = loss + effort_weight * torch.mean(torch.diff(controls) ** 2)
    return loss


def surrogate_health(results: SimulationResults) -> dict[str, float]:
    """How well the RBF surrogate tracked the plant over an episode.

    ``prediction_std`` near zero means the basis functions have saturated and the
    model is emitting a constant, which carries no gradient — the failure mode
    that makes surrogate-based training stop without any error being raised.
    """
    if not results.rbf_predictions:
        return {}
    predicted = torch.stack([p.detach().reshape(-1)[0] for p in results.rbf_predictions])
    actual = torch.stack([p.detach().reshape(-1)[0] for p in results.positions])
    return {
        "rmse": float(torch.sqrt(torch.mean((predicted - actual) ** 2))),
        "prediction_std": float(predicted.std()),
        "plant_std": float(actual.std()),
    }


def run_episode(
    system: BaseSystem,
    pid: PID,
    simulation_config: SimulationConfig,
    extract_rbf_input: RbfExtractor,
    extract_lstm_input: LstmExtractor,
    rbf_model: nn.Module,
    lstm_model: Optional[nn.Module] = None,
    loss_function: LossFn = tracking_loss,
    session: Session = "train",
    optimizer: Optional[Optimizer] = None,
    grad_clip: float | None = 1.0,
    verbose: bool = False,
) -> EpisodeReport:
    """Run one episode; train the LSTM in place when ``session == "train"``.

    Sessions:
        ``train``      LSTM active, TBPTT updates applied.
        ``validation`` LSTM active, no updates (runs under ``no_grad``).
        ``static``     LSTM inactive: the PID keeps its initial gains. This is
                       the fixed-gain control baseline.
    """
    if session == "train" and optimizer is None:
        raise ValueError("Optimizer must be provided for a training session.")
    if session == "static" and lstm_model is not None:
        raise ValueError("A static session must not be given an LSTM model.")

    dt = torch.as_tensor(simulation_config.dt, dtype=torch.float32)
    max_dt = system.min_dt()
    if not bool(dt < max_dt):
        raise ValueError(
            f"Time step {float(dt):.4g}s is too large for this plant "
            f"(needs < {float(max_dt):.4g}s) — the integration would not resolve "
            "its dynamics."
        )

    training = session == "train"
    gain_scale = simulation_config.gain_scale
    report = EpisodeReport(results=SimulationResults())
    results = report.results
    hidden = None
    window_start = 0
    window = max(1, simulation_config.tbptt_window)

    # Keep gradients only where they are used: validation and static runs are
    # pure forward passes and should not build a graph at all.
    grad_context = torch.enable_grad() if training else torch.no_grad()

    with grad_context:
        for step in range(simulation_config.num_steps):
            setpoint = torch.as_tensor(
                simulation_config.setpoints[step], dtype=torch.float32
            ).reshape(-1)[0]

            # ── controller gains ─────────────────────────────────────────
            if lstm_model is not None and step >= simulation_config.warm_up_steps:
                lstm_input = extract_lstm_input(simulation_config, results)
                normalised_gains, hidden = lstm_model(lstm_input, hidden)
                kp, ki, kd = normalised_gains[0] * gain_scale
                pid.update_gains(kp, ki, kd)
            else:
                kp, ki, kd = pid.gains

            # ── control law ──────────────────────────────────────────────
            error = setpoint - system.X
            control_output = pid.compute(error, dt, measurement=system.X)

            # ── surrogate prediction of the *next* output ────────────────
            rbf_prediction = rbf_model(extract_rbf_input(system, control_output))
            rbf_prediction = rbf_prediction.reshape(-1)[0]

            # ── plant ────────────────────────────────────────────────────
            disturbance = simulation_config.disturbance_at(step)
            system.apply_control(control_output, disturbance)

            # ── record ───────────────────────────────────────────────────
            previous_error = (
                results.error_history[-1] if results.error_history else torch.tensor(0.0)
            )
            results.time_points.append(step * dt)
            results.setpoints.append(setpoint)
            results.positions.append(system.X.reshape(-1)[0])
            results.control_outputs.append(control_output.reshape(-1)[0])
            results.rbf_predictions.append(rbf_prediction)
            results.error_history.append(error.reshape(-1)[0])
            results.error_diff_history.append(error.reshape(-1)[0] - previous_error)
            results.kp_values.append(torch.as_tensor(kp).reshape(-1)[0])
            results.ki_values.append(torch.as_tensor(ki).reshape(-1)[0])
            results.kd_values.append(torch.as_tensor(kd).reshape(-1)[0])
            results.disturbances.append(disturbance.reshape(-1)[0])

            # ── truncated backprop through time ──────────────────────────
            reached_window_end = (step + 1 - window_start) >= window
            past_warm_up = step >= simulation_config.warm_up_steps
            if training and reached_window_end and past_warm_up:
                loss = loss_function(results, simulation_config, window_start, step + 1)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    lstm_model.parameters(),
                    grad_clip if grad_clip is not None else float("inf"),
                )
                optimizer.step()

                report.window_losses.append(float(loss.detach()))
                report.grad_norms.append(float(grad_norm))
                results.losses.append(float(loss.detach()))
                if verbose:
                    print(
                        f"  step {step + 1:>4}  loss {float(loss):.5f}  "
                        f"|grad| {float(grad_norm):.3e}"
                    )

                # Truncate the graph: keep the numbers, drop the history.
                system.detach_state()
                pid.detach_state()
                hidden = _detach_hidden(hidden)
                results.detach_all()
                window_start = step + 1

    return report


def _detach_hidden(hidden):
    if hidden is None:
        return None
    return tuple(h.detach() for h in hidden)


def run_simulation(
    system: BaseSystem,
    pid: PID,
    simulation_config: SimulationConfig,
    extract_rbf_input: RbfExtractor,
    extract_lstm_input: LstmExtractor,
    rbf_model: nn.Module,
    lstm_model: Optional[nn.Module] = None,
    **kwargs,
) -> SimulationResults:
    """Backwards-compatible wrapper returning only the trajectory."""
    return run_episode(
        system=system,
        pid=pid,
        simulation_config=simulation_config,
        extract_rbf_input=extract_rbf_input,
        extract_lstm_input=extract_lstm_input,
        rbf_model=rbf_model,
        lstm_model=lstm_model,
        **kwargs,
    ).results


def train_rbf_model(
    model: SystemRBFModel,
    X: torch.Tensor,
    y: torch.Tensor,
    num_epochs: int = 500,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    optimizer: Literal["adam", "sgd"] = "adam",
    gradient_clip_value: float | None = None,
    validation_split: float = 0.2,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Fit the surrogate model, holding out a validation split.

    Returns train and validation loss curves. Without the held-out split there
    is no way to tell a surrogate that generalises from one that memorised its
    sample points — and the controller is trained *through* this model, so a
    surrogate that only fits its training set produces confidently wrong
    gradients.
    """
    criterion = nn.MSELoss()
    if optimizer == "adam":
        opt = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        opt = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Invalid optimizer type: {optimizer!r}")

    n_val = int(len(X) * validation_split)
    shuffle = torch.randperm(len(X))
    val_idx, train_idx = shuffle[:n_val], shuffle[n_val:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    history: dict[str, list[float]] = {"train": [], "val": []}

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, len(X_train), batch_size):
            batch = permutation[i : i + batch_size]
            loss = criterion(model(X_train[batch]), y_train[batch])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if gradient_clip_value is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            opt.step()
            epoch_losses.append(float(loss.detach()))

        history["train"].append(sum(epoch_losses) / len(epoch_losses))

        model.eval()
        with torch.no_grad():
            history["val"].append(
                float(criterion(model(X_val), y_val).detach()) if n_val else float("nan")
            )

        if verbose and (epoch + 1) % max(1, num_epochs // 10) == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}]  "
                f"train {history['train'][-1]:.5f}  val {history['val'][-1]:.5f}"
            )

    return history
