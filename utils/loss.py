import torch
from classes.simulation import SimulationConfig, SimulationResults


def default_loss(results: SimulationResults, config: SimulationConfig, step: int) -> torch.Tensor:
    left_slice = max(0, step - config.sequence_length)
    right_slice = step

    # Slices
    positions = results.rbf_predictions[left_slice:right_slice:config.sequence_step]
    setpoints = results.setpoints[left_slice:right_slice:config.sequence_step]

    # Tensors
    positions_tensor = torch.stack(positions)
    setpoints_tensor = torch.stack(setpoints)

    # Errors
    tracking_error = torch.mean((positions_tensor - setpoints_tensor) ** 2)
    overshoot = torch.mean(torch.relu(positions_tensor - setpoints_tensor))

    loss = (
        0.5 * tracking_error +
        0.7 * overshoot
    )
    return loss
