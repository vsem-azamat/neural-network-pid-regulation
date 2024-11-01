import torch
from classes.simulation import SimulationConfig, SimulationResults


def custom_loss(results: SimulationResults, config: SimulationConfig, step: int) -> torch.Tensor:
    left_slice = max(0, step - config.sequence_length)
    right_slice = step

    # Slices
    positions = results.rbf_predictions[left_slice:right_slice:config.sequence_step]
    setpoints = results.setpoints[left_slice:right_slice:config.sequence_step]
    kp_values = results.kp_values[left_slice:right_slice:config.sequence_step]
    # ki_values = results.ki_values[left_slice:right_slice:config.sequence_step]
    # kd_values = results.kd_values[left_slice:right_slice:config.sequence_step]


    # Tensors
    positions_tensor = torch.stack(positions)
    setpoints_tensor = torch.stack(setpoints)
    kp_tensor = torch.stack(kp_values)
    # ki_tensor = torch.stack(ki_values)
    # kd_tensor = torch.stack(kd_values)

    # Errors
    tracking_error = torch.mean((positions_tensor - setpoints_tensor) ** 2)
    overshoot = torch.mean(torch.relu(positions_tensor - setpoints_tensor))
    kp_gain = torch.mean(kp_tensor**2)
    # ki_gain = torch.mean(ki_tensor**2)
    # kd_gain = torch.mean(kd_tensor**2)

    loss = (
        0.5 * tracking_error +
        0.7 * overshoot
        # 0.2 * kp_gain
        # 0.1 * ki_gain + 
        # 0.1 * kd_gain
    )
    return loss
