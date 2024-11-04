import os
import matplotlib.pyplot as plt
from typing import Literal

from config import cnfg
from utils.run import SimulationResults

class DynamicPlot:
    def __init__(self, system_name='<System>'):
        self.system_name = system_name
        self.fig, self.axs = plt.subplots(4, 2, figsize=(20, 30))
        self.fig.suptitle(f'Adaptive LSTM-PID {system_name} Control Simulation', fontsize=16)
        self.colors = [
            'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
            'darkblue', 'darkorange', 'darkgreen', 'darkred', '#800080', '#ff00ff', '#00ffff', '#ffff00',
            '#00ff00', '#ff0000', '#0000ff', '#000000', '#808080', '#800000', '#008000', '#000080', '#808000',
            '#800080', '#008080', '#c0c0c0', '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#00ffff', '#ff00ff',
            '#800000', '#008000', '#000080', '#808000', '#800080', '#008080', '#c0c0c0', '#ff0000', '#00ff00',
            '#0000ff', '#ffff00', '#00ffff', '#ff00ff', '#800000', '#008000', '#000080', '#808000', '#800080',
            
        ]
        self.epoch_count = 0

    def update_plot(self, results: SimulationResults, label: str, session: Literal["train", "validation"]) -> None:
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
        alpha = self.epoch_count / (self.epoch_count + 1)

        if session == "train":
            col = 0
        else:
            col = 1

        # Plot positions
        self.axs[0, col].plot(time_points, positions, label=label, alpha=alpha, color=color)
        self.axs[0, col].plot(time_points, positions_rbf, linestyle=':', alpha=alpha, color=color)
        self.axs[0, col].plot(time_points, setpoints, linestyle='--', color='red', alpha=alpha)
        self.axs[0, col].set_ylabel('Position')
        self.axs[0, col].set_title(f'{session.capitalize()}: {self.system_name} Position')
        self.axs[0, col].legend(loc='upper right')
        self.axs[0, col].grid()

        # Plot control outputs
        self.axs[1, col].plot(time_points, control_outputs, alpha=alpha, color=color)
        self.axs[1, col].set_ylabel('Control Output')
        self.axs[1, col].set_title(f'{session.capitalize()}: {self.system_name} Control Output')
        self.axs[1, col].grid()

        # Plot PID parameters
        self.axs[2, col].plot(time_points, kp_values, alpha=alpha, color=color, linestyle='--')
        self.axs[2, col].plot(time_points, ki_values, alpha=alpha, color=color, linestyle='-.')
        self.axs[2, col].plot(time_points, kd_values, alpha=alpha, color=color, linestyle=':')
        self.axs[2, col].set_ylabel('PID Parameters')
        self.axs[2, col].set_title(f'{session.capitalize()}: {self.system_name} PID Parameters')
        self.axs[2, col].legend(['Kp --', 'Ki -.', 'Kd :'])
        self.axs[2, col].grid()

        # Plot losses
        if len(losses) > 0:
            self.axs[3, col].plot(range(len(losses)), losses, alpha=alpha, color=color)
            self.axs[3, col].set_xlabel('Training Steps')
            self.axs[3, col].set_ylabel('Loss')
            self.axs[3, col].set_title(f'{session.capitalize()}: {self.system_name} Loss')
            self.axs[3, col].grid()

        # Plot errors
        if len(errors) > 0:
            self.axs[3, 1].plot(time_points, errors, alpha=alpha, color=color)
            self.axs[3, 1].set_xlabel('Time')
            self.axs[3, 1].set_ylabel('Error')
            self.axs[3, 1].set_title(f'{session.capitalize()}: {self.system_name} Error')
            self.axs[3, 1].grid()

        # Refresh the plot
        plt.draw()

    def save(self, save_name: str) -> None:
        if save_name is not None:
            save_path = os.path.join(cnfg.PLOTS_DIR, save_name + f'_epoch_{self.epoch_count}.png')
            self.fig.savefig(save_path)
            print(f"Plot saved as {save_name}_epoch_{self.epoch_count}.png")

    def show(self) -> None:
        plt.show()
