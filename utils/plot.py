import os
import matplotlib.pyplot as plt
from typing import Literal

from config import cnfg
from classes.simulation import SimulationResults, LearningConfig


class DynamicPlot:
    def __init__(self, system_name: str = "<System>") -> None:
        self.system_name = system_name

        # Create a figure with 5 rows and 3 columns (train, validation, static)
        self.fig, self.axs = plt.subplots(5, 3, figsize=(30, 35))
        self.fig.suptitle(
            f"Adaptive LSTM-PID {system_name} Control Simulation", fontsize=16
        )
        self.colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]
        self.epoch_count = 0

        # Adjust spacing between subplots
        self.fig.subplots_adjust(wspace=0.4, hspace=0.6)

        # Initialize legends trackers
        self.legend_added = {
            "position": False,
            "control_output": False,
            "pid_parameters": False,
            "loss": False,
            "error": False,
        }

    def update_plot(
        self,
        results: SimulationResults,
        label: str,
        session: Literal["train", "validation", "static"],
    ) -> None:
        self.epoch_count += 1

        # Convert results to numpy arrays
        results = results.to_numpy()
        time_points = results.time_points
        positions = results.positions
        positions_rbf = results.rbf_predictions
        setpoints = results.setpoints
        control_outputs = results.control_outputs
        kp_values = results.kp_values
        ki_values = results.ki_values
        kd_values = results.kd_values
        losses = results.losses
        errors = results.error_history

        # Choose color for the current epoch
        color = self.colors[self.epoch_count % len(self.colors)]
        alpha = 0.9  # Higher alpha for better visibility
        line_width = 1.5  # Increased line width

        # Map session to column index
        session_to_col = {"train": 0, "validation": 1, "static": 2}
        col = session_to_col.get(session, 2)  # Default to 'static' column

        # Plot positions in the first row
        ax_position = self.axs[0, col]
        # Plot actual positions
        ax_position.plot(
            time_points,
            positions,
            label=f"Actual {self.system_name}",
            alpha=alpha,
            color=color,
            linewidth=line_width,
        )
        # Plot RBF predictions
        ax_position.plot(
            time_points,
            positions_rbf,
            linestyle=":",
            alpha=alpha,
            color=color,
            linewidth=line_width,
        )
        # Plot setpoints (only once)
        if not self.legend_added["position"]:
            ax_position.plot(
                time_points,
                setpoints,
                linestyle="--",
                color="red",
                alpha=0.7,
                linewidth=line_width,
                label="Setpoint",
            )
        else:
            ax_position.plot(
                time_points,
                setpoints,
                linestyle="--",
                color="red",
                alpha=0.7,
                linewidth=line_width,
            )
        ax_position.set_ylabel("Position", fontsize=12)
        ax_position.set_title(
            f"{session.capitalize()}: {self.system_name} Position", fontsize=14
        )
        # Add legend only once
        if not self.legend_added["position"]:
            ax_position.legend(loc="upper right", fontsize=10)
            self.legend_added["position"] = True
        ax_position.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot control outputs in the second row
        ax_control = self.axs[1, col]
        ax_control.plot(
            time_points,
            control_outputs,
            alpha=alpha,
            color=color,
            linewidth=line_width,
            label=f"Forced {self.system_name}",
        )
        ax_control.set_ylabel("Control Output", fontsize=12)
        ax_control.set_title(
            f"{session.capitalize()}: {self.system_name} Control Output", fontsize=14
        )
        # Add legend only once
        if not self.legend_added["control_output"]:
            ax_control.legend(loc="upper right", fontsize=10)
            self.legend_added["control_output"] = True
        ax_control.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot PID parameters in the third row
        ax_pid = self.axs[2, col]
        # Plot Kp, Ki, Kd with same color but different styles
        ax_pid.plot(
            time_points,
            kp_values,
            alpha=alpha,
            color=color,
            linestyle="--",
            linewidth=line_width,
            label=f"Kp (Epoch {self.epoch_count})",
        )
        ax_pid.plot(
            time_points,
            ki_values,
            alpha=alpha,
            color=color,
            linestyle="-.",
            linewidth=line_width,
            label=f"Ki (Epoch {self.epoch_count})",
        )
        ax_pid.plot(
            time_points,
            kd_values,
            alpha=alpha,
            color=color,
            linestyle=":",
            linewidth=line_width,
            label=f"Kd (Epoch {self.epoch_count})",
        )
        ax_pid.set_ylabel("PID Parameters", fontsize=12)
        ax_pid.set_title(
            f"{session.capitalize()}: {self.system_name} PID Parameters", fontsize=14
        )
        # Add legend only once, showing line styles
        if not self.legend_added["pid_parameters"]:
            # Create custom legend handles
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D([0], [0], color="black", linestyle="--", label="Kp"),
                Line2D([0], [0], color="black", linestyle="-.", label="Ki"),
                Line2D([0], [0], color="black", linestyle=":", label="Kd"),
            ]
            ax_pid.legend(handles=legend_elements, fontsize=10)
            self.legend_added["pid_parameters"] = True
        ax_pid.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot losses in the fourth row (only for 'train' and 'validation')
        ax_loss = self.axs[3, col]
        if session in ["train", "validation"] and len(losses) > 0:
            ax_loss.plot(
                range(len(losses)),
                losses,
                alpha=alpha,
                color=color,
                linewidth=line_width,
                label=f"Epoch {self.epoch_count}",
            )
            ax_loss.set_xlabel("Training Steps", fontsize=12)
            ax_loss.set_ylabel("Loss", fontsize=12)
            ax_loss.set_title(
                f"{session.capitalize()}: {self.system_name} Loss", fontsize=14
            )
            # Add legend only once
            if not self.legend_added["loss"]:
                ax_loss.legend(loc="upper right", fontsize=10)
                self.legend_added["loss"] = True
            ax_loss.grid(True, which="both", linestyle="--", linewidth=0.5)
        else:
            # Hide the subplot if there are no losses to plot
            ax_loss.set_visible(False)

        # Plot errors in the fifth row
        ax_error = self.axs[4, col]
        if len(errors) > 0:
            ax_error.plot(
                time_points,
                errors,
                alpha=alpha,
                color=color,
                linewidth=line_width,
                label=f"Epoch {self.epoch_count}",
            )
            ax_error.set_xlabel("Time", fontsize=12)
            ax_error.set_ylabel("Error", fontsize=12)
            ax_error.set_title(
                f"{session.capitalize()}: {self.system_name} Error", fontsize=14
            )
            # Add legend only once
            if not self.legend_added["error"]:
                ax_error.legend(loc="upper right", fontsize=10)
                self.legend_added["error"] = True
            ax_error.grid(True, which="both", linestyle="--", linewidth=0.5)
        else:
            # Hide the subplot if there are no errors to plot
            ax_error.set_visible(False)

        # Adjust layout to leave space for the main title
        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

        # Redraw the plot
        plt.draw()

    def save(self, save_name: str, learning_config: LearningConfig) -> None:
        if save_name:
            formatted_name = f"{save_name}_ep_{learning_config.num_epochs}_lr_{learning_config.learning_rate}.png"
            save_path = os.path.join(
                "plots", formatted_name
            )  # Assuming 'plots' directory
            self.fig.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Plot saved to {save_path}")

    def show(self) -> None:
        plt.show()


def plot_rbf_training_results(
    control_inputs,
    Y_rbf,
    Y_actual,
    losses: list[float],
    system_name="<System>",
    state_label="Pozice",
    num_epochs: int | None = None,
    learning_rate: float | None = None,
    optimizer_name: str | None = None,
) -> None:
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(
        control_inputs,
        Y_rbf,
        label="RBF Model",
        marker="o",
        linestyle="-",
        markersize=3,
    )
    axs[0].plot(
        control_inputs,
        Y_actual,
        label="Skutečný systém",
        marker="x",
        linestyle="-",
        markersize=3,
    )
    axs[0].set_title(f"Srovnání RBF modelu a skutečného systému {system_name}")
    axs[0].set_xlabel("Řídicí vstup")
    axs[0].set_ylabel(state_label)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(range(len(losses)), losses)
    axs[1].set_title(f"Tréninkové ztráty pro RBF model {system_name}")
    axs[1].set_xlabel("Tréninkové kroky")
    axs[1].set_ylabel("Ztráta")
    axs[1].grid(True)

    plt.tight_layout()
    save_name = f"rbf_{system_name.lower()}_ep_{num_epochs}_lr_{learning_rate}_opt_{optimizer_name}"
    save_path = os.path.join(cnfg.LEARNING_PLOTS, f"{save_name}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
