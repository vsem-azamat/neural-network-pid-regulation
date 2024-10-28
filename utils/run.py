import torch
from typing import Literal, Callable
from torch.optim.optimizer import Optimizer

from entities.pid import PID
from entities.systems import BaseSystem
from utils.others import calculate_angle_2p
from classes.simulation import SimulationConfig, SimulationResults


def simulation_step(
    system: BaseSystem,
    pid: PID,
    lstm_model: torch.nn.Module,
    rbf_model: torch.nn.Module,
    simulation_config: SimulationConfig,
    results: SimulationResults,
    step: int,
    hidden: torch.Tensor | None = None
):
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
    input_array = torch.zeros(4, simulation_config.sequence_length)

    # Populate input array with historical data
    error_history_len = min(simulation_config.sequence_length, len(results.error_history))
    kp_values_len = min(simulation_config.sequence_length, len(results.kp_values))
    ki_values_len = min(simulation_config.sequence_length, len(results.ki_values))
    kd_values_len = min(simulation_config.sequence_length, len(results.kd_values))

    # Paste last values
    input_array[0, -error_history_len:] = torch.tensor(
        results.error_history[-error_history_len:] if results.error_history else [0.0] * simulation_config.sequence_length
    )
    input_array[1, -kp_values_len:] = torch.tensor(
        results.kp_values[-kp_values_len:] if results.kp_values else [0.0] * simulation_config.sequence_length
    )
    input_array[2, -ki_values_len:] = torch.tensor(
        results.ki_values[-ki_values_len:] if results.ki_values else [0.0] * simulation_config.sequence_length
    )
    input_array[3, -kd_values_len:] = torch.tensor(
        results.kd_values[-kd_values_len:] if results.kd_values else [0.0] * simulation_config.sequence_length
    )

    # Prepare LSTM input
    lstm_input = input_array.transpose(0, 1).unsqueeze(0)

    if step > 10:
        lstm_pred, hidden = lstm_model(
            lstm_input, 
            hidden
        )
        kp, ki, kd = lstm_pred[0] * 5
    else:
        kp, ki, kd = torch.tensor([3, 0.1, 1.0]).unbind(0)

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

    results.error_history.append(error)

    # >>> Calculate: Angle for HISTORY and LOSS <<<
    if step < 2 and simulation_config.sequence_step > step * 2:
        results.angle_history.append(torch.tensor(0.0))
        results.losses.append(torch.tensor(0.0))
    else:
        angle = calculate_angle_2p(
            (results.time_points[-2], results.positions[-2]), 
            (results.time_points[-1], results.positions[-1])
        )
        results.angle_history.append(angle)

    return hidden


def run_simulation(
    system: BaseSystem,
    pid: PID,
    lstm_model: torch.nn.Module,
    rbf_model: torch.nn.Module,
    simulation_config: SimulationConfig,
    loss_function: Callable[[SimulationResults, SimulationConfig, int], torch.Tensor],
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
        hidden = simulation_step(
            system,
            pid,
            lstm_model,
            rbf_model,
            simulation_config,
            results,
            step,
            hidden
        )

        if step > 2:
            loss = loss_function(
                results,
                simulation_config,
                step,
            )
        else:
            loss = torch.tensor(0.0)
        results.losses.append(loss)

        # >>> Calculate: Loss and Update Optimizer <<<
        if session == 'train' and \
            step % simulation_config.sequence_length == 0 and \
            step >= simulation_config.sequence_length:
                        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Step: {step}, Loss: {loss.item()}")

    return results

# For validation, you can now call run_simulation with session='validation' and no optimizer
def run_validation(
    system: BaseSystem,
    pid: PID,
    lstm_model: torch.nn.Module,
    rbf_model: torch.nn.Module,
    simulation_config: SimulationConfig,
    loss_function: Callable[[SimulationResults, SimulationConfig, int], torch.Tensor]
):
    return run_simulation(
        system,
        pid,
        lstm_model,
        rbf_model,
        simulation_config,
        session='validation',
        optimizer=None,
        loss_function=loss_function
    )
