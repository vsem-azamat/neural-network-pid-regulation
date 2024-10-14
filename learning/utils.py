import os
import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import matplotlib.pyplot as plt

from config import cnfg
from entities.pid import PID
from entities.systems import BaseSystem


def tensor2list(tensor) -> list:
    try:
        return [t.item() for t in tensor]
    except AttributeError:
        return [t for t in tensor]


def calculate_angle_2p(pos1, pos2):
    return torch.atan((pos2[1] - pos1[1]) / (pos2[0] - pos1[0])) * 180 / np.pi


def custom_loss(positions, setpoints, control_output, pid_params, time_points, alpha=0.7, beta=1, gamma=0.2, delta=0.1):
    tracking_error = torch.mean((positions - setpoints) ** 2)
    params_regularization = torch.mean(pid_params ** 2)
    control_effort = torch.mean(control_output ** 2)
    
    # Calculate movement direction changes
    direction_changes = 0
    if len(positions) > 2:
        prev_angle = calculate_angle_2p((time_points[0], positions[0]), 
                                        (time_points[1], positions[1]))
        for i in range(2, len(positions)):
            current_angle = calculate_angle_2p((time_points[i-1], positions[i-1]), 
                                               (time_points[i], positions[i]))
            angle_change = abs(current_angle - prev_angle)
            direction_changes += angle_change
            prev_angle = current_angle
    
    direction_penalty = direction_changes / (len(positions) - 2) if len(positions) > 2 else 0
    
    # Overshoot penalty
    overshoot = torch.mean(torch.relu(positions - setpoints))
    angle_penalty = torch.arctan((positions[-1] - positions[-2]) / (time_points[-1] - time_points[-2])) ** 2
    
    # print(tracking_error)
    return (
        alpha * tracking_error +
        # control_effort +
        beta * params_regularization +
        gamma * angle_penalty +
        gamma * direction_penalty +
        delta * overshoot
        )


def plot_simulation_results(epoch_results, validation_results, setpoint_train, setpoint_val, system_name = '<System>', save_name = None):
    fig, axs = plt.subplots(5, 2, figsize=(20, 30))
    fig.suptitle(f'Adaptive LSTM-PID {system_name} Control Simulation', fontsize=16)

    # Plot training results (left column)
    for epoch, results in enumerate(epoch_results):
        time_points, positions, control_outputs, kp_values, ki_values, kd_values, angle_history, losses = results
        time_points, positions, control_outputs, kp_values, ki_values, kd_values, angle_history, losses = tensor2list(time_points), tensor2list(positions), tensor2list(control_outputs), tensor2list(kp_values), tensor2list(ki_values), tensor2list(kd_values), tensor2list(angle_history), tensor2list(losses)
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
    time_points_val, positions_val, control_outputs_val, kp_val, ki_val, kd_val, angle_history_val = tensor2list(time_points_val), tensor2list(positions_val), tensor2list(control_outputs_val), tensor2list(kp_val), tensor2list(ki_val), tensor2list(kd_val), tensor2list(angle_history_val)

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


def run_simulation(
    system: BaseSystem, 
    pid: PID, 
    lstm_model: torch.nn.Module, 
    rbf_model, 
    setpoints, 
    steps, 
    dt,
    optimizer: Optimizer = None,
    train=True, 
    validation=False,
    sequence_length=100
    ):

    error_history = []
    error_diff_history = []
    time_points, control_outputs = [], []
    positions, rbf_predictions = [], []
    kp_values, ki_values, kd_values = [], [], []
    angle_history = []
    losses = []
    hidden = None

    current_setpoint_idx = 0
    steps_per_setpoint = steps // len(setpoints)

    for step in range(steps):
        current_time = step * dt.item()
        
        # Change setpoint at intervals
        if step % steps_per_setpoint == 0 and step > 0:
            current_setpoint_idx = (current_setpoint_idx + 1) % len(setpoints)
        
        setpoint = setpoints[current_setpoint_idx]
        current_position = system.X
        error = setpoint - current_position
        error_diff = (error - error_history[-1])/dt if len(error_history) > 0 else torch.tensor(0.0)

        # Prepare the input for the RBF model
        rbf_input = torch.tensor([current_position.item(), system.dXdT.item(), system.d2XdT2.item(), 0.0]).unsqueeze(0)
        rbf_pred = rbf_model(rbf_input)

        # Combine the error and RBF prediction as the LSTM input
        lstm_input = torch.tensor([error.item(), error_diff.item(), rbf_pred[0].item()]).unsqueeze(0).unsqueeze(0)

        pid_params, hidden = lstm_model(lstm_input, hidden)
        kp, ki, kd = pid_params[0] * 5

        pid.update_gains(kp, ki, kd)
        control_output = pid.compute(error, dt)
        system.apply_control(control_output)

        time_points.append(current_time)
        positions.append(current_position)
        control_outputs.append(control_output.item())

        rbf_predictions.append(rbf_pred)
        error_history.append(error.item())
        error_diff_history.append(error_diff.item())

        kp_values.append(kp)
        ki_values.append(ki)
        kd_values.append(kd)

        if len(positions) >= 2:
            angle = calculate_angle_2p((time_points[-2], positions[-2]), (time_points[-1], positions[-1]))
            angle_history.append(angle)

        if train and step % sequence_length == 0 and step > sequence_length:
            optimizer.zero_grad()
            sequence_length_min = min(sequence_length, len(error_history))
            
            # Loss calculation
            loss_rbf_positions = torch.cat(rbf_predictions[-sequence_length_min:], dim=0)
            # print(rbf_predictions[-sequence_length_min:])
            # print(loss_rbf_positions[:5])
            loss_setpoints = setpoints[current_setpoint_idx].repeat(sequence_length_min)
            loss_control_output = torch.tensor(control_outputs[-sequence_length_min:])
            loss_time_points = time_points[-sequence_length_min:]
            loss_pid_params = torch.cat((
                torch.tensor(kp_values[-sequence_length_min:]).unsqueeze(1),
                torch.tensor(ki_values[-sequence_length_min:]).unsqueeze(1),
                torch.tensor(kd_values[-sequence_length_min:]).unsqueeze(1)
            ))

            loss = custom_loss(
                positions=loss_rbf_positions,
                setpoints=loss_setpoints,
                control_output=loss_control_output,
                pid_params=loss_pid_params,
                time_points=loss_time_points

            )
            losses.append(loss.item())
            
            loss.backward()
            optimizer.step()

    if validation:
        return time_points, positions, control_outputs, kp_values, ki_values, kd_values, angle_history, []

    return time_points, positions, control_outputs, kp_values, ki_values, kd_values, angle_history, losses
