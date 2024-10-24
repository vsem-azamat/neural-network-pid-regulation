import torch


def custom_loss(
    dt: torch.Tensor,
    positions: list[torch.Tensor], 
    setpoints: list[torch.Tensor], 
    control_outputs: list[torch.Tensor],
    kp_values: list[torch.Tensor],
    ki_values: list[torch.Tensor],
    kd_values: list[torch.Tensor],
    # pid_params: list[torch.Tensor],  # Each element is tensor([kp, ki, kd])
    alpha=0.5, beta=0.7, gamma=1.0, delta=1.0, epsilon=0.1, zeta=0.01, eta=0.02
    ):
    # Convert lists to tensors
    positions_tensor = torch.stack(positions)
    setpoints_tensor = torch.stack(setpoints)
    # kp_values_tensor = torch.stack(kp_values)
    # ki_values_tensor = torch.stack(ki_values)
    # kd_values_tensor = torch.stack(kd_values)
    # control_outputs_tensor = torch.stack()
    # pid_params_tensor = torch.stack(pid_params)
    
    # # Tracking error
    tracking_error = torch.mean((positions_tensor - setpoints_tensor) ** 2)
    
    # # Overshoot penalty
    overshoot = torch.mean(torch.relu(positions_tensor - setpoints_tensor))
    
    # # Reverse initial movement penalty
    # initial_error = setpoints_tensor[0] - positions_tensor[0]
    # initial_rate_of_change = positions_tensor[1] - positions_tensor[0]
    # reverse_movement_penalty = torch.relu(-initial_error * initial_rate_of_change)
    
    # # Stability penalty (control smoothness)
    # control_output_derivative = control_outputs_tensor[1:] - control_outputs_tensor[:-1]
    # control_smoothness = torch.mean(control_output_derivative ** 2)
    # kp_smoothness = torch.mean(kp_values_tensor[1:] - kp_values_tensor[:-1])
    # ki_smoothness = torch.mean(ki_values_tensor[1:] - ki_values_tensor[:-1])
    # kd_smoothness = torch.mean(kd_values_tensor[1:] - kd_values_tensor[:-1])
    
    # # Control effort penalty
    # control_effort = torch.mean(control_outputs_tensor ** 2)
    
    # # PID parameters regularization
    # params_regularization = torch.mean(pid_params_tensor ** 2)

    # Angle penalty
    angle = torch.atan2(positions_tensor[1:] - positions_tensor[:-1], dt)
    angle_penalty = torch.mean(angle ** 2)
    
    # Total loss
    loss = (
        alpha * tracking_error +
        beta * overshoot +
        # gamma * reverse_movement_penalty +
        # delta * control_smoothness +
        # gamma * kp_smoothness +
        # gamma * ki_smoothness +
        # gamma * kd_smoothness +
        # epsilon * control_effort +
        # zeta * params_regularization
        eta * angle_penalty
    )
    return loss
