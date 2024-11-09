import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from utils.normalizer import Normalizer
from models.sys_rbf import SystemRBFModel
from entities.systems.pendulum import NonLinearPendulumCart


def generate_training_data(pendulum_cart, num_samples=2000):
    X = torch.zeros((num_samples, 5))  # [x, x_dot, theta, theta_dot, control_input]
    y = torch.zeros(
        (num_samples, 4)
    )  # [next_x, next_x_dot, next_theta, next_theta_dot]

    for i in range(num_samples):
        x = torch.rand(1) * 4 - 2  # Random position between -2 and 2
        x_dot = torch.rand(1) * 4 - 2  # Random velocity between -2 and 2
        theta = torch.rand(1) * np.pi - np.pi / 2  # Random angle between -pi/2 and pi/2
        theta_dot = torch.rand(1) * 4 - 2  # Random angular velocity between -2 and 2
        control_input = (
            torch.rand(1) * 20 - 10
        )  # Random control input between -10 and 10

        pendulum_cart.x = x
        pendulum_cart.x_dot = x_dot
        pendulum_cart.theta = theta
        pendulum_cart.theta_dot = theta_dot
        next_state = pendulum_cart.apply_control(control_input)

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


def train_rbf_model(
    model, pendulum_cart, num_epochs=500, batch_size=32, learning_rate=0.001
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X, y = generate_training_data(pendulum_cart)

    # Normalize the data
    X_normalizer = Normalizer(X)
    y_normalizer = Normalizer(y)

    X_normalized = X_normalizer.normalize(X)
    y_normalized = y_normalizer.normalize(y)

    losses = []

    for epoch in range(num_epochs):
        epoch_losses = []
        for i in range(0, len(X_normalized), batch_size):
            batch_X = X_normalized[i : i + batch_size]
            batch_y = y_normalized[i : i + batch_size]

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return losses, X_normalizer, y_normalizer


def plot_training_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.show()


def compare_predictions(
    model, pendulum_cart, X_normalizer, y_normalizer, num_steps=200
):
    initial_state = torch.tensor(
        [0.5, 0.0, 0.1, 0.0]
    )  # Initial state: x, x_dot, theta, theta_dot
    control_inputs = torch.linspace(-5, 5, num_steps)

    rbf_states = []
    actual_states = []

    (
        pendulum_cart.x,
        pendulum_cart.x_dot,
        pendulum_cart.theta,
        pendulum_cart.theta_dot,
    ) = initial_state

    for control in control_inputs:
        # RBF model prediction
        with torch.no_grad():
            rbf_input = torch.cat([pendulum_cart.get_state(), control.unsqueeze(0)])
            rbf_input_normalized = X_normalizer.normalize(rbf_input.unsqueeze(0))
            rbf_next_state_normalized = model(rbf_input_normalized)
            rbf_next_state = y_normalizer.denormalize(
                rbf_next_state_normalized
            ).squeeze(0)
            rbf_states.append(rbf_next_state)

        # Actual pendulum cart system
        actual_next_state = pendulum_cart.apply_control(control)
        actual_states.append(actual_next_state)

        # Update pendulum cart state for next iteration
        (
            pendulum_cart.x,
            pendulum_cart.x_dot,
            pendulum_cart.theta,
            pendulum_cart.theta_dot,
        ) = actual_next_state

    return control_inputs.numpy(), torch.stack(rbf_states), torch.stack(actual_states)


def plot_comparison(control_inputs, rbf_states, actual_states):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    state_labels = [
        "Cart Position",
        "Cart Velocity",
        "Pendulum Angle",
        "Pendulum Angular Velocity",
    ]

    for i in range(4):
        row = i // 2
        col = i % 2
        axs[row, col].plot(
            control_inputs,
            rbf_states[:, i],
            label="RBF Model",
            marker="o",
            linestyle="-",
            markersize=3,
        )
        axs[row, col].plot(
            control_inputs,
            actual_states[:, i],
            label="Actual System",
            marker="x",
            linestyle="-",
            markersize=3,
        )
        axs[row, col].set_title(f"{state_labels[i]}")
        axs[row, col].set_xlabel("Control Input")
        axs[row, col].set_ylabel("State Value")
        axs[row, col].legend()
        axs[row, col].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Initialize the NonLinearPendulumCart system
    pendulum_cart = NonLinearPendulumCart(
        cart_mass=torch.tensor(1.0),
        pendulum_mass=torch.tensor(0.1),
        pendulum_length=torch.tensor(1.0),
        friction=torch.tensor(0.1),
        gravity=torch.tensor(9.81),
        dt=torch.tensor(0.01),
    )

    # Initialize and train the RBF model
    rbf_model = SystemRBFModel(hidden_features=50, input_size=5, output_size=4)
    losses, X_normalizer, y_normalizer = train_rbf_model(
        rbf_model, pendulum_cart, num_epochs=500, learning_rate=0.001
    )

    # Save the trained model and normalizers
    from utils.save_load import save_model, save_pickle

    save_model(rbf_model, "sys_rbf_pendulum.pth")
    save_pickle((X_normalizer, y_normalizer), "sys_rbf_pendulum_normalizers.pkl")

    # Plot training loss
    plot_training_loss(losses)

    # Compare predictions
    control_inputs, rbf_states, actual_states = compare_predictions(
        rbf_model, pendulum_cart, X_normalizer, y_normalizer
    )

    # Plot comparison
    plot_comparison(control_inputs, rbf_states, actual_states)

    # Calculate and print Mean Squared Error for each state variable
    mse = torch.mean((rbf_states - actual_states) ** 2, dim=0)
    for i, state_label in enumerate(
        [
            "Cart Position",
            "Cart Velocity",
            "Pendulum Angle",
            "Pendulum Angular Velocity",
        ]
    ):
        print(f"Mean Squared Error for {state_label}: {mse[i].item():.6f}")
