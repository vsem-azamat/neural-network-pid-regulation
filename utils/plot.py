import os
import matplotlib.pyplot as plt
from typing import Literal

from config import cnfg
from utils.run import SimulationResults


class DynamicPlot:
    def __init__(self, system_name='<System>'):
        self.system_name = system_name
        self.fig, self.axs = plt.subplots(4, 3, figsize=(30, 30))  # Adjusted to 3 columns
        self.fig.suptitle(f'Adaptive LSTM-PID {system_name} Control Simulation', fontsize=20)
        self.colors = [
            'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 
            'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
        ]
        self.epoch_count = 0
        plt.subplots_adjust(wspace=0.4, hspace=0.6)  # Adding spacing between subplots

    def update_plot(self, results: SimulationResults, label: str, session: Literal["train", "validation", "static"]) -> None:
        self.epoch_count += 1
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

        color = self.colors[self.epoch_count % len(self.colors)]
        alpha = 0.9  # Higher alpha for better visibility
        line_width = 1.5  # Increased line width

        if session == "train":
            col = 0
        elif session == "validation":
            col = 1
        else:  # "static"
            col = 2

        # Plot positions
        self.axs[0, col].plot(time_points, positions, label=label, alpha=alpha, color=color, linewidth=line_width)
        self.axs[0, col].plot(time_points, positions_rbf, linestyle=':', alpha=alpha, color=color, linewidth=line_width)
        self.axs[0, col].plot(time_points, setpoints, linestyle='--', color='red', alpha=0.7, linewidth=line_width)
        self.axs[0, col].set_ylabel('Position', fontsize=14)
        self.axs[0, col].set_title(f'{session.capitalize()}: {self.system_name} Position', fontsize=16)
        self.axs[0, col].legend(loc='upper right', fontsize=10)
        self.axs[0, col].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plot control outputs
        self.axs[1, col].plot(time_points, control_outputs, alpha=alpha, color=color, linewidth=line_width)
        self.axs[1, col].set_ylabel('Control Output', fontsize=14)
        self.axs[1, col].set_title(f'{session.capitalize()}: {self.system_name} Control Output', fontsize=16)
        self.axs[1, col].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plot PID parameters
        self.axs[2, col].plot(time_points, kp_values, alpha=alpha, color=color, linestyle='--', linewidth=line_width)
        self.axs[2, col].plot(time_points, ki_values, alpha=alpha, color=color, linestyle='-.', linewidth=line_width)
        self.axs[2, col].plot(time_points, kd_values, alpha=alpha, color=color, linestyle=':', linewidth=line_width)
        self.axs[2, col].set_ylabel('PID Parameters', fontsize=14)
        self.axs[2, col].set_title(f'{session.capitalize()}: {self.system_name} PID Parameters', fontsize=16)
        self.axs[2, col].legend(['Kp --', 'Ki -.', 'Kd :'], fontsize=10)
        self.axs[2, col].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plot losses (only for train and validation)
        if session != "static" and len(losses) > 0:
            self.axs[3, col].plot(range(len(losses)), losses, alpha=alpha, color=color, linewidth=line_width)
            self.axs[3, col].set_xlabel('Training Steps', fontsize=14)
            self.axs[3, col].set_ylabel('Loss', fontsize=14)
            self.axs[3, col].set_title(f'{session.capitalize()}: {self.system_name} Loss', fontsize=16)
            self.axs[3, col].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Plot errors
        if len(errors) > 0:
            self.axs[3, 2].plot(time_points, errors, alpha=alpha, color=color, linewidth=line_width)
            self.axs[3, 2].set_xlabel('Time', fontsize=14)
            self.axs[3, 2].set_ylabel('Error', fontsize=14)
            self.axs[3, 2].set_title(f'{session.capitalize()}: {self.system_name} Error', fontsize=16)
            self.axs[3, 2].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Refresh the plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for the main title
        plt.draw()

    def save(self, save_name: str) -> None:
        if save_name is not None:
            save_path = os.path.join(cnfg.PLOTS_DIR, save_name + f'_epoch_{self.epoch_count}.png')
            self.fig.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Plot saved as {save_name}_epoch_{self.epoch_count}.png")

    def show(self) -> None:
        plt.show()
