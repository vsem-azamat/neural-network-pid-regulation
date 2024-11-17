import torch
import numpy as np

from models.sys_rbf import SystemRBFModel
from entities.systems import NonLinearPendulumCart
from utils import save_load
from utils.run import train_rbf_model
from utils.plot import plot_rbf_training_results


def generate_training_data(
    pendulum_cart: NonLinearPendulumCart, num_samples: int = 2000
):
    X = torch.zeros((num_samples, 5))  # [x, x_dot, theta, theta_dot, control_input]
    y = torch.zeros(
        (num_samples, 4)
    )  # [next_x, next_x_dot, next_theta, next_theta_dot]

    for i in range(num_samples):
        x = torch.rand(1) * 4 - 2  # [-2, 2] meters
        x_dot = torch.rand(1) * 20 - 10  # [-10, 10] m/s
        theta = torch.rand(1) * 2 * torch.pi - torch.pi  # [-pi, pi] radians
        theta_dot = torch.rand(1) * 20 - 10  # [-10, 10] rad/s
        control_input = torch.rand(1) * 200 - 100  # [-10, 10] N

        # Set the system's state
        pendulum_cart.x = x
        pendulum_cart.x_dot = x_dot
        pendulum_cart.theta = theta
        pendulum_cart.theta_dot = theta_dot

        # Apply control to get next state
        next_state = pendulum_cart.apply_control(control_input)

        # Store input and output
        X[i] = torch.tensor(
            [
                x.item(),
                x_dot.item(),
                theta.item(),
                theta_dot.item(),
                control_input.item(),
            ]
        )
        y[i] = next_state

    return X, y


def compare_predictions(model, pendulum_cart: NonLinearPendulumCart, num_steps=200):
    # Initial conditions
    initial_x = torch.tensor(0.0)
    initial_x_dot = torch.tensor(0.0)
    initial_theta = torch.tensor(0.1)  # Small angle
    initial_theta_dot = torch.tensor(0.0)
    control_inputs = torch.zeros(num_steps)  # Zero control force

    rbf_states = []
    actual_states = []

    # Set initial state
    pendulum_cart.x = initial_x
    pendulum_cart.x_dot = initial_x_dot
    pendulum_cart.theta = initial_theta
    pendulum_cart.theta_dot = initial_theta_dot

    for control in control_inputs:
        # RBF model prediction
        with torch.no_grad():
            rbf_input = torch.tensor(
                [
                    [
                        pendulum_cart.x.item(),
                        pendulum_cart.x_dot.item(),
                        pendulum_cart.theta.item(),
                        pendulum_cart.theta_dot.item(),
                        control.item(),
                    ]
                ]
            )
            rbf_next_state = model(rbf_input).numpy()[0]
            rbf_states.append(rbf_next_state)

        # Actual system
        actual_next_state = pendulum_cart.apply_control(control)
        actual_states.append(actual_next_state.numpy())

        # Update pendulum_cart state for next iteration
        pendulum_cart.x = actual_next_state[0]
        pendulum_cart.x_dot = actual_next_state[1]
        pendulum_cart.theta = actual_next_state[2]
        pendulum_cart.theta_dot = actual_next_state[3]

    return (
        np.array(control_inputs),
        np.array(rbf_states),
        np.array(actual_states),
    )


if __name__ == "__main__":
    # Initialize the NonLinearPendulumCart system
    pendulum_cart = NonLinearPendulumCart(
        cart_mass=torch.tensor(1.0),
        pendulum_mass=torch.tensor(0.1),
        pendulum_length=torch.tensor(0.5),
        friction=torch.tensor(0.1),
        gravity=torch.tensor(9.81),
        dt=torch.tensor(0.01),
    )

    # Generate training data
    X, y = generate_training_data(pendulum_cart)

    # Compute mean and std for normalization
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0, unbiased=False)
    y_mean = y.mean(dim=0)
    y_std = y.std(dim=0, unbiased=False)

    # Initialize and train the RBF model
    rbf_model = SystemRBFModel(
        input_mean=X_mean,
        input_std=X_std,
        output_mean=y_mean,
        output_std=y_std,
        input_size=5,  # Number of input features
        output_size=4,  # Number of output features
        hidden_features=50,
    )

    # Training settings
    lr = 0.001
    optimizer_name = "adam"
    gradient_clip_value = None
    num_epochs = 600
    batch_size = 32

    losses = train_rbf_model(
        rbf_model,
        X,
        y,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=lr,
    )

    # Save the trained model (including the normalizers as part of the model)
    save_load.save_rbf_model(rbf_model, "sys_rbf_pendulum_cart.pth")

    # Compare predictions
    control_inputs, rbf_states, actual_states = compare_predictions(
        rbf_model, pendulum_cart
    )

    # Plot comparison
    plot_rbf_training_results(
        control_inputs,
        rbf_states[:, 2],  # Pendulum angle from RBF model
        actual_states[:, 2],  # Pendulum angle from actual system
        losses,
        system_name="NonLinearPendulumCart",
        state_label="Pendulum Angle (rad)",
        num_epochs=num_epochs,
        learning_rate=lr,
        optimizer_name=optimizer_name,
    )

    # Calculate and print Mean Squared Error
    mse = np.mean((rbf_states - actual_states) ** 2)
    print(f"Mean Squared Error between RBF model and actual system: {mse:.6f}")
