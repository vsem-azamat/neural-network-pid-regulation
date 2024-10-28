import os
import matplotlib.pyplot as plt

from config import cnfg
from utils.run import SimulationResults, SimulationConfig


def plot_simulation_results(
    training_results: list[SimulationResults],
    training_config: SimulationConfig,

    validation_result: SimulationResults,
    validation_config: SimulationConfig,

    system_name: str = '<System>', 
    save_name=None
    ):

    fig, axs = plt.subplots(4, 2, figsize=(20, 30))
    fig.suptitle(f'Adaptive LSTM-PID {system_name} Control Simulation', fontsize=16)

    colors = [
        'blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
        'darkblue', 'darkorange', 'darkgreen', 'darkred', '#800080', '#ff00ff', '#00ffff', '#ffff00',
        '#00ff00', '#ff0000', '#0000ff', '#000000', '#808080', '#800000', '#008000', '#000080', '#808000',
        '#800080', '#008080', '#c0c0c0', '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#00ffff', '#ff00ff',
        '#800000', '#008000', '#000080', '#808000', '#800080', '#008080', '#c0c0c0', '#ff0000', '#00ff00',
    ]
    # Plot training results (left column)
    for epoch_idx, results in enumerate(training_results):
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


        alpha = (epoch_idx + 1) / len(training_results)

        # Plot positions
        axs[0, 0].plot(time_points, positions, label=f'Epoch {epoch_idx+1}', alpha=alpha, color=colors[epoch_idx])
        axs[0, 0].plot(time_points, positions_rbf, linestyle=':', alpha=alpha, color=colors[epoch_idx])
        axs[0, 0].plot(time_points, setpoints, linestyle='--', color='red', alpha=alpha)
        axs[0, 0].set_ylabel('Position')
        axs[0, 0].set_title(f'Training: {system_name} Position')
        if epoch_idx == len(training_results) - 1:
            axs[0, 0].legend()
        axs[0, 0].grid()

        # Plot control outputs
        axs[1, 0].plot(time_points, control_outputs, alpha=alpha)
        axs[1, 0].set_ylabel('Control Output')
        axs[1, 0].set_title(f'Training: {system_name} Control Output')
        axs[1, 0].grid()

        # Plot PID parameters
        axs[2, 0].plot(time_points, kp_values, alpha=alpha, color=colors[epoch_idx], linestyle='--')
        axs[2, 0].plot(time_points, ki_values, alpha=alpha, color=colors[epoch_idx], linestyle='-.')
        axs[2, 0].plot(time_points, kd_values, alpha=alpha, color=colors[epoch_idx], linestyle=':')
        axs[2, 0].set_ylabel('PID Parameters')
        axs[2, 0].set_title(f'Training: {system_name} PID Parameters')
        if epoch_idx == len(training_results) - 1:
            axs[2, 0].legend(['Kp --', 'Ki -.', 'Kd :'])
        axs[2, 0].grid()


        # Plot losses
        if len(losses) > 0:
            axs[3, 0].plot(range(len(losses)), losses, alpha=alpha)
            axs[3, 0].set_xlabel('Training Steps')
            axs[3, 0].set_ylabel('Loss')
            axs[3, 0].set_title(f'Training: {system_name} Loss')
            axs[3, 0].grid()

        # Plot errors
        if len(errors) > 0:
            axs[3, 1].plot(time_points, errors, alpha=alpha)
            axs[3, 1].set_xlabel('Time')
            axs[3, 1].set_ylabel('Error')
            axs[3, 1].set_title(f'Training: {system_name} Error')
            axs[3, 1].grid()
            


    # Plot validation results (right column)
    validation_result = validation_result.to_numpy()
    time_points = validation_result.time_points
    positions = validation_result.positions
    positions_rbf = validation_result.rbf_predictions
    setpoints = validation_result.setpoints
    control_outputs = validation_result.control_outputs
    kp_values = validation_result.kp_values
    ki_values = validation_result.ki_values
    kd_values = validation_result.kd_values
    losses = validation_result.losses

    # Plot positions
    axs[0, 1].plot(time_points, positions, label=f'{system_name} Position')
    axs[0, 1].plot(time_points, positions_rbf, linestyle=':', label=f'{system_name} RBF Prediction')
    axs[0, 1].plot(time_points, setpoints, linestyle='--', color='red', label='Setpoint')
    axs[0, 1].set_ylabel('Position')
    axs[0, 1].set_title(f'Validation: {system_name} Position')
    axs[0, 1].legend()
    axs[0, 1].grid()

    # Plot control outputs
    axs[1, 1].plot(time_points, control_outputs)
    axs[1, 1].set_ylabel('Control Output')
    axs[1, 1].set_title(f'Validation: {system_name} Control Output')
    axs[1, 1].grid()

    # Plot PID parameters
    axs[2, 1].plot(time_points, kp_values, label='Kp')
    axs[2, 1].plot(time_points, ki_values, label='Ki')
    axs[2, 1].plot(time_points, kd_values, label='Kd')
    axs[2, 1].set_ylabel('PID Parameters')
    axs[2, 1].set_title(f'Validation: {system_name} PID Parameters')
    axs[2, 1].legend()
    axs[2, 1].grid()

    plt.tight_layout()
    plt.show()

    if save_name is not None:
        save_path = os.path.join(cnfg.PLOTS_DIR, save_name + '.png')
        fig.savefig(save_path)
        print(f"Plot saved as {save_name}")
