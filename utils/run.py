import torch
from typing import Literal, Callable
from torch.optim.optimizer import Optimizer

from entities.pid import PID
from entities.systems import BaseSystem
from utils.loss import default_loss
from utils.others import calculate_angle_2p
from classes.simulation import SimulationConfig, SimulationResults


def simulation_step(
    system: BaseSystem,
    pid: PID,
    lstm_model: torch.nn.Module | None,
    rbf_model: torch.nn.Module,
    simulation_config: SimulationConfig,
    results: SimulationResults,
    step: int,
    extract_rbf_input: Callable[[BaseSystem, SimulationResults], torch.Tensor],
    extract_lstm_input: Callable[[SimulationConfig, SimulationResults], torch.Tensor],
    hidden: torch.Tensor | None = None
):
    current_time = step * simulation_config.dt

    # >>> RBF <<<
    rbf_input = extract_rbf_input(system, results)
    rbf_pred = rbf_model(rbf_input)[0][0]

    # >>> LSTM <<<
    lstm_input = extract_lstm_input(simulation_config, results)

    if step > 10 and lstm_model is not None:
        lstm_pred, hidden = lstm_model(
            lstm_input, 
            hidden
        )
        kp, ki, kd = lstm_pred[0] * 5
        pid.update_gains(kp, ki, kd)
    else:
        kp, ki, kd = pid.Kp.clone().detach(), pid.Ki.clone().detach(), pid.Kd.clone().detach()

    # >>> PID <<<
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
    simulation_config: SimulationConfig,
    extract_rbf_input: Callable[[BaseSystem, SimulationResults], torch.Tensor],
    extract_lstm_input: Callable[[SimulationConfig, SimulationResults], torch.Tensor],
    rbf_model: torch.nn.Module,
    lstm_model: torch.nn.Module | None = None,
    loss_function: Callable[[SimulationResults, SimulationConfig, int], torch.Tensor] = default_loss,
    session: Literal['train', 'validation', 'static'] = 'train',
    optimizer: Optimizer | None = None,
):
    # >>> Asserts <<<
    if session == 'train' and optimizer is None:
        raise ValueError("Optimizer must be provided for training session.")

    if session in ['validation', 'static'] and optimizer is not None:
        print("Optimizer is not needed for validation/static session.")

    torch.autograd.set_detect_anomaly(True)

    hidden = None
    steps = len(simulation_config.setpoints)
    results = SimulationResults()
    for step in range(steps):
        hidden = simulation_step(
            system=system,
            pid=pid,
            lstm_model=lstm_model,
            rbf_model=rbf_model,
            simulation_config=simulation_config,
            results=results,
            step=step,
            hidden=hidden,
            extract_rbf_input=extract_rbf_input,
            extract_lstm_input=extract_lstm_input,
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

