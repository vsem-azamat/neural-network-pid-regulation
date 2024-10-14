from math import pi
import torch
from typing import Literal
from torch.optim.optimizer import Optimizer

from entities.pid import PID
from entities.systems import BaseSystem
from utils.loss import custom_loss
from utils.others import calculate_angle_2p
from classes.simulation import SimulationConfig, SimulationResults


def run_simulation(
    system: BaseSystem,
    pid: PID,
    lstm_model: torch.nn.Module,
    rbf_model: torch.nn.Module,
    simulation_config: SimulationConfig,
    session: Literal['train', 'validation'] = 'train',
    optimizer: Optimizer | None = None,
):
    # >>> Asserts <<<
    if session == 'train' and optimizer is None:
        raise ValueError("Optimizer must be provided for training session.")

    if session == 'validation' and optimizer is not None:
        print("Optimizer is not needed for validation session.")

    torch.autograd.set_detect_anomaly(True)

    hidden = None
    steps = len(simulation_config.setpoints)
    results = SimulationResults()
    for step in range(steps):
        current_time = step * simulation_config.dt
        
        # >>> RBF <<<
        rbf_input = torch.tensor([
            system.X, 
            system.dXdT, 
            system.d2XdT2, 
            results.control_outputs[-1] if results.control_outputs else 0.0
        ]).unsqueeze(0)
        rbf_pred = rbf_model(rbf_input)[0][0]

        # >>> LSTM <<<
        input_array = torch.zeros(3, simulation_config.input_sequence_length)
        
        # Populate input array with historical data
        error_history_len = min(simulation_config.input_sequence_length, len(results.error_history))
        error_diff_history_len = min(simulation_config.input_sequence_length, len(results.error_diff_history))
        rbf_predictions_len = min(simulation_config.input_sequence_length, len(results.rbf_predictions))
        
        # Paste last values
        input_array[0, -error_history_len:] = torch.tensor(results.error_history[-error_history_len:] if results.error_history else [0.0] * simulation_config.input_sequence_length)
        input_array[1, -error_diff_history_len:] = torch.tensor(results.error_diff_history[-error_diff_history_len:] if results.error_diff_history else [0.0] * simulation_config.input_sequence_length)
        input_array[2, -rbf_predictions_len:] = torch.tensor(results.rbf_predictions[-rbf_predictions_len:] if results.rbf_predictions else [0.0] * simulation_config.input_sequence_length)

        # Prepare LSTM input
        lstm_input = input_array.transpose(0, 1).unsqueeze(0)

        lstm_pred, hidden = lstm_model(
            lstm_input, 
            hidden
            )
        kp, ki, kd = lstm_pred[0] * 5

        # >>> PID <<<
        pid.update_gains(kp, ki, kd)
        error = (simulation_config.setpoints[step] - system.X)
        control_output = pid.compute(error, simulation_config.dt)
        system.apply_control(control_output)

        # >>> Update results <<<
        results.setpoints.append(simulation_config.setpoints[step])
        results.time_points.append(current_time)
        results.positions.append(system.X)
        results.control_outputs.append(control_output)
        results.rbf_predictions.append(rbf_pred)

        results.kp_values.append(kp)
        results.ki_values.append(ki)
        results.kd_values.append(kd)
        results.pid_params.append(lstm_pred)

        results.error_history.append(error)
        
        # >>> Calculate: Angle for HISTORY and LOSS <<<
        if step < 2 and simulation_config.input_steps > step * 2:
            results.angle_history.append(torch.tensor(0.0))
            results.losses.append(torch.tensor(0.0))
            continue
        angle = calculate_angle_2p(
            (results.time_points[-2], results.positions[-2]), 
            (results.time_points[-1], results.positions[-1])
        )
        results.angle_history.append(angle)

        left_slice = max(0, step - simulation_config.sequence_length)
        right_slice = step
        loss = custom_loss(
            dt=simulation_config.dt,
            positions=results.rbf_predictions[left_slice:right_slice:simulation_config.input_steps],
            setpoints=results.setpoints[left_slice:right_slice:simulation_config.input_steps],
            control_outputs=results.control_outputs[left_slice:right_slice:simulation_config.input_steps], # TODO: probleim with 'inplace operation'
            # pid_params=[
            #     torch.stack([results.kp_values[step], results.ki_values[step], results.kd_values[step]])
            #     for step in range(left_slice, right_slice)
            # ]
        )
        results.losses.append(loss)

        # >>> Calculate: Loss <<<
        if session == 'train' and \
            step % simulation_config.sequence_length == 0 and \
            step >= simulation_config.sequence_length:
                        
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            print(f"Step: {step}, Loss: {loss.item()}")

    return results
