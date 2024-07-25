import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from config import cnfg

def calculate_angle_2p(pos1, pos2) -> np.float64:
    return np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0]) * 180 / np.pi


def custom_loss(position, setpoint, control_output, pid_params, time_points, alpha=0.7, beta=0.2, gamma=0.2, delta=0.1):
    tracking_error = torch.mean((position - setpoint) ** 2)
    control_effort = torch.mean(control_output ** 2)
    params_regularization = torch.mean(pid_params ** 2)
    
    # Calculate movement direction changes
    # direction_changes = 0
    # if len(position) > 2:
    #     prev_angle = calculate_angle_2p((time_points[0], position[0]), 
    #                                     (time_points[1], position[1]))
    #     for i in range(2, len(position)):
    #         current_angle = calculate_angle_2p((time_points[i-1], position[i-1]), 
    #                                            (time_points[i], position[i]))
    #         angle_change = abs(current_angle - prev_angle)
    #         direction_changes += angle_change
    #         prev_angle = current_angle
    
    # direction_penalty = direction_changes / (len(position) - 2) if len(position) > 2 else 0
    
    # Overshoot penalty
    # overshoot = torch.mean(torch.relu(position - setpoint))
    angle_penalty = torch.arctan((position[-1] - position[-2]) / (time_points[-1] - time_points[-2])) ** 2
    
    return (
        alpha * tracking_error +
        # (1 - alpha - gamma - delta) * control_effort + 
        beta * params_regularization + 
        gamma * angle_penalty
        # gamma * direction_penalty 
        # delta * overshoot
        )


def plot_simulation_results(epoch_results, validation_results, setpoint_train, setpoint_val, system_name = '<System>', save_name = None):
    fig, axs = plt.subplots(5, 2, figsize=(20, 30))
    fig.suptitle(f'Adaptive LSTM-PID {system_name} Control Simulation', fontsize=16)

    # Plot training results (left column)
    for epoch, results in enumerate(epoch_results):
        time_points, positions, control_outputs, kp_values, ki_values, kd_values, angle_history, losses = results
        alpha = (epoch + 1) / len(epoch_results)
        
        axs[0, 0].plot(time_points, positions, label=f'Epoch {epoch+1}', alpha=alpha)
        axs[1, 0].plot(time_points, control_outputs, alpha=alpha)
        axs[2, 0].plot(time_points, kp_values, alpha=alpha)
        axs[2, 0].plot(time_points, ki_values, alpha=alpha)
        axs[2, 0].plot(time_points, kd_values, alpha=alpha)
        axs[3, 0].plot(time_points[1:], angle_history, alpha=alpha)
        axs[4, 0].plot(range(len(losses)), losses, alpha=alpha)

    axs[0, 0].axhline(y=setpoint_train.item(), color='r', linestyle='--', label='Setpoint')
    axs[0, 0].set_ylabel('Position')
    axs[0, 0].set_title(f'Training: {system_name} Position')
    axs[0, 0].legend()
    axs[0, 0].grid()

    axs[1, 0].set_ylabel('Control Output')
    axs[1, 0].set_title(f'Training: {system_name} Control Output')
    axs[1, 0].grid()

    axs[2, 0].set_ylabel('PID Parameters')
    axs[2, 0].set_title(f'Training: {system_name} PID Parameters')
    axs[2, 0].grid()

    axs[3, 0].set_ylabel('Angle (degrees)')
    axs[3, 0].set_title(f'Training: {system_name} Angle History')
    axs[3, 0].grid()

    axs[4, 0].set_xlabel('Training Steps')
    axs[4, 0].set_ylabel('Loss')
    axs[4, 0].set_title(f'Training: {system_name} Loss')
    axs[4, 0].grid()

    # Plot validation results (right column)
    time_points_val, positions_val, control_outputs_val, kp_val, ki_val, kd_val, angle_history_val, _ = validation_results

    axs[0, 1].plot(time_points_val, positions_val, label=f'{system_name} Position')
    axs[0, 1].axhline(y=setpoint_val.item(), color='r', linestyle='--', label='Setpoint')
    axs[0, 1].set_ylabel('Position')
    axs[0, 1].set_title(f'Validation: {system_name} Position')
    axs[0, 1].legend()
    axs[0, 1].grid()

    axs[1, 1].plot(time_points_val, control_outputs_val, label='Control Output')
    axs[1, 1].set_ylabel('Control Output')
    axs[1, 1].set_title(f'Validation: {system_name} Control Output')
    axs[1, 1].legend()
    axs[1, 1].grid()

    axs[2, 1].plot(time_points_val, kp_val, label='Kp')
    axs[2, 1].plot(time_points_val, ki_val, label='Ki')
    axs[2, 1].plot(time_points_val, kd_val, label='Kd')
    axs[2, 1].set_ylabel('PID Parameters')
    axs[2, 1].set_title(f'Validation: {system_name} PID Parameters')
    axs[2, 1].legend()
    axs[2, 1].grid()

    axs[3, 1].plot(time_points_val[1:], angle_history_val, label='Angle')
    axs[3, 1].set_xlabel('Time')
    axs[3, 1].set_ylabel('Angle (degrees)')
    axs[3, 1].set_title(f'Validation: {system_name} Angle History')
    axs[3, 1].legend()
    axs[3, 1].grid()

    plt.tight_layout()
    plt.show()

    if save_name is not None:
        fig.savefig(os.path.join(cnfg.PUBLIC_DIR, save_name + '.png'))
        print(f"Plot saved as {save_name}")
