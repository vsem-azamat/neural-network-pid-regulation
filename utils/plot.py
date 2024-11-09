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
            f"Adaptive LSTM-PID {system_name} Control Simulation", fontsize=20
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
        ax_position.plot(
            time_points,
            positions,
            label=f"{label} Actual",
            alpha=alpha,
            color=color,
            linewidth=line_width,
        )
        ax_position.plot(
            time_points,
            positions_rbf,
            linestyle=":",
            alpha=alpha,
            color="black",
            linewidth=line_width,
            label=f"{label} RBF Prediction",
        )
        ax_position.plot(
            time_points,
            setpoints,
            linestyle="--",
            color="red",
            alpha=0.7,
            linewidth=line_width,
            label="Setpoint",
        )
        ax_position.set_ylabel("Position", fontsize=14)
        ax_position.set_title(
            f"{session.capitalize()}: {self.system_name} Position", fontsize=16
        )
        ax_position.legend(loc="upper right", fontsize=10)
        ax_position.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot control outputs in the second row
        ax_control = self.axs[1, col]
        ax_control.plot(
            time_points, control_outputs, alpha=alpha, color=color, linewidth=line_width
        )
        ax_control.set_ylabel("Control Output", fontsize=14)
        ax_control.set_title(
            f"{session.capitalize()}: {self.system_name} Control Output", fontsize=16
        )
        ax_control.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Plot PID parameters in the third row
        ax_pid = self.axs[2, col]
        ax_pid.plot(
            time_points,
            kp_values,
            alpha=alpha,
            color="red",
            linestyle="--",
            linewidth=line_width,
            label="Kp",
        )
        ax_pid.plot(
            time_points,
            ki_values,
            alpha=alpha,
            color="green",
            linestyle="-.",
            linewidth=line_width,
            label="Ki",
        )
        ax_pid.plot(
            time_points,
            kd_values,
            alpha=alpha,
            color="blue",
            linestyle=":",
            linewidth=line_width,
            label="Kd",
        )
        ax_pid.set_ylabel("PID Parameters", fontsize=14)
        ax_pid.set_title(
            f"{session.capitalize()}: {self.system_name} PID Parameters", fontsize=16
        )
        ax_pid.legend(fontsize=10)
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
            )
            ax_loss.set_xlabel("Training Steps", fontsize=14)
            ax_loss.set_ylabel("Loss", fontsize=14)
            ax_loss.set_title(
                f"{session.capitalize()}: {self.system_name} Loss", fontsize=16
            )
            ax_loss.grid(True, which="both", linestyle="--", linewidth=0.5)
        else:
            # Hide the subplot if there are no losses to plot
            ax_loss.set_visible(False)

        # Plot errors in the fifth row
        ax_error = self.axs[4, col]
        if len(errors) > 0:
            ax_error.plot(
                time_points, errors, alpha=alpha, color=color, linewidth=line_width
            )
            ax_error.set_xlabel("Time", fontsize=14)
            ax_error.set_ylabel("Error", fontsize=14)
            ax_error.set_title(
                f"{session.capitalize()}: {self.system_name} Error", fontsize=16
            )
            ax_error.grid(True, which="both", linestyle="--", linewidth=0.5)
        else:
            # Hide the subplot if there are no errors to plot
            ax_error.set_visible(False)

        # Adjust layout to leave space for the main title
        plt.tight_layout(rect=(0., 0.03, 1., 0.95))

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
) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))
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
        label="Actual System",
        marker="x",
        linestyle="-",
        markersize=3,
    )
    axs[0].set_title(f"Comparison of RBF Model vs Actual {system_name} System")
    axs[0].set_xlabel("Control Input")
    axs[0].set_ylabel("Position")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(range(len(losses)), losses)
    axs[1].set_title(f"Training Losses for {system_name} RBF Model")
    axs[1].set_xlabel("Training Steps")
    axs[1].set_ylabel("Loss")
    axs[1].grid(True)

    plt.tight_layout()
    save_name = f"rbf_{system_name.lower()}"
    save_path = os.path.join(cnfg.PLOTS_DIR, f"{save_name}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
