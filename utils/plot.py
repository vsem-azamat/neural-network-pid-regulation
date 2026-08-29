"""Plotting for training runs and episode trajectories."""

import os

import matplotlib.pyplot as plt

from config import cnfg

def plot_training_history(
    history: list[dict],
    system_name: str = "<System>",
    show: bool = False,
) -> None:
    """Loss, gradient norm and tracking error over training episodes.

    The gradient-norm panel is not decoration: a flat line at zero here is
    exactly what the original project produced for ten epochs while printing a
    plausible-looking loss curve.
    """
    epochs = [h["epoch"] for h in history]
    fig, axs = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    axs[0].plot(epochs, [h["loss"] for h in history], color="tab:blue")
    axs[0].set_ylabel("Window loss")
    axs[0].set_title(f"{system_name}: LSTM gain scheduler training")

    axs[1].semilogy(epochs, [h["grad_norm"] for h in history], color="tab:red")
    axs[1].set_ylabel("|grad| (log)")

    axs[2].plot(epochs, [h["tracking_iae"] for h in history], color="tab:green")
    axs[2].set_ylabel("Episode IAE")
    axs[2].set_xlabel("Episode")

    for ax in axs:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(cnfg.LEARNING_PLOTS, f"lstm_{system_name.lower()}_history.png")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Plot saved to {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_episode(
    results,
    system_name: str = "<System>",
    label: str = "",
    filename: str | None = None,
    show: bool = False,
) -> None:
    """Reference tracking, control signal and scheduled gains for one episode."""
    numeric = results.to_numpy()
    time = numeric.time_points

    fig, axs = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    axs[0].plot(time, numeric.setpoints, "r--", linewidth=1.5, label="Setpoint")
    axs[0].plot(time, numeric.positions, linewidth=1.8, label="Plant output")
    if numeric.rbf_predictions:
        axs[0].plot(
            time, numeric.rbf_predictions, ":", linewidth=1.2, label="RBF surrogate"
        )
    axs[0].set_ylabel("Output")
    axs[0].set_title(f"{system_name} {label}".strip())
    axs[0].legend(loc="best")

    axs[1].plot(time, numeric.control_outputs, color="tab:red", linewidth=1.5,
                label="Control")
    if any(float(d) != 0.0 for d in numeric.disturbances):
        axs[1].plot(time, numeric.disturbances, color="tab:gray", linewidth=1.0,
                    alpha=0.8, label="Disturbance")
    axs[1].set_ylabel("Control signal")
    axs[1].legend(loc="best")

    axs[2].plot(time, numeric.kp_values, label="Kp", linewidth=1.5)
    axs[2].plot(time, numeric.ki_values, label="Ki", linewidth=1.5)
    axs[2].plot(time, numeric.kd_values, label="Kd", linewidth=1.5)
    axs[2].set_ylabel("Gain")
    axs[2].set_xlabel("Time (s)")
    axs[2].legend(loc="best")

    for ax in axs:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        path = os.path.join(cnfg.LEARNING_PLOTS, filename)
        plt.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Plot saved to {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_rbf_training_results(
    control_inputs,
    Y_rbf,
    Y_actual,
    history: dict[str, list[float]] | list[float],
    system_name: str = "<System>",
    state_label: str = "Output",
    num_epochs: int | None = None,
    learning_rate: float | None = None,
    optimizer_name: str | None = None,
    show: bool = False,
) -> None:
    """Surrogate accuracy over the actuator range, plus the training curves."""
    if not isinstance(history, dict):
        history = {"train": list(history), "val": []}

    fig, axs = plt.subplots(2, 1, figsize=(10, 9))

    axs[0].plot(control_inputs, Y_actual, label="True plant", linewidth=2)
    axs[0].plot(control_inputs, Y_rbf, label="RBF surrogate", linestyle="--", linewidth=2)
    axs[0].set_title(f"{system_name}: one-step-ahead prediction on held-out excitation")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel(state_label)
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(history["train"], label="Train")
    if history.get("val"):
        axs[1].plot(history["val"], label="Validation", linestyle="--")
    axs[1].set_yscale("log")
    axs[1].set_title(f"{system_name}: RBF training loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("MSE (log scale)")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_name = (
        f"rbf_{system_name.lower()}_ep_{num_epochs}"
        f"_lr_{learning_rate}_opt_{optimizer_name}"
    )
    save_path = os.path.join(cnfg.LEARNING_PLOTS, f"{save_name}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)
