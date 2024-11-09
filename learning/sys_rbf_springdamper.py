import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from models.sys_rbf import SystemRBFModel
from entities.systems.springdamper import SpringDamper
from utils import save_load


def generate_training_data(msd: SpringDamper, num_samples: int = 1000):
    X = torch.zeros(
        (num_samples, 4)
    )  # [position, velocity, acceleration, control_input]
    y = torch.zeros((num_samples, 1))  # next_position

    for i in range(num_samples):
        position = torch.rand(1) * 20 - 10  # [-10, 10]
        velocity = torch.rand(1) * 20 - 10  # [-10, 10]
        acceleration = torch.rand(1) * 20 - 10  # [-10, 10]
        control_input = torch.rand(1) * 20 - 10  # [-10, 10]

        # Set the state of the system
        msd.position = position
        msd.velocity = velocity
        msd.acceleration = acceleration

        # Apply control input and get the next position
        next_position = msd.apply_control(control_input)

        X[i] = torch.tensor(
            [
                position.item(),
                velocity.item(),
                acceleration.item(),
                control_input.item(),
            ]
        )
        y[i] = next_position  # Ensure next_position is correctly computed

    return X, y


def train_rbf_model(model, X, y, num_epochs=500, batch_size=64, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(num_epochs):
        epoch_losses = []
        permutation = torch.randperm(X.size()[0])
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]
        for i in range(0, len(X), batch_size):
            batch_X = X_shuffled[i : i + batch_size]
            batch_y = y_shuffled[i : i + batch_size]

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

    return losses


def plot_training_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.show()


def compare_predictions(model, msd: SpringDamper, num_steps=200):
    initial_position = torch.tensor(0.0)
    initial_velocity = torch.tensor(0.0)
    initial_acceleration = torch.tensor(0.0)
    control_inputs = torch.linspace(-1, 1, num_steps)

    rbf_positions = []
    actual_positions = []

    msd.position = initial_position
    msd.velocity = initial_velocity
    msd.acceleration = initial_acceleration

    for control in control_inputs:
        # RBF model prediction
        with torch.no_grad():
            rbf_input = torch.tensor(
                [[msd.X.item(), msd.dXdT.item(), msd.d2XdT2.item(), control.item()]]
            )
            rbf_next_position = model(rbf_input).item()
            rbf_positions.append(rbf_next_position)

        # Actual MassSpringDamper system
        actual_next_position = msd.apply_control(control).item()
        actual_positions.append(actual_next_position)

        # Update MassSpringDamper state for next iteration
        msd.position = torch.tensor(actual_next_position)

    return control_inputs.numpy(), rbf_positions, actual_positions


def plot_comparison(control_inputs, rbf_positions, actual_positions):
    plt.figure(figsize=(12, 6))
    plt.plot(
        control_inputs,
        rbf_positions,
        label="RBF Model",
        marker="o",
        linestyle="-",
        markersize=3,
    )
    plt.plot(
        control_inputs,
        actual_positions,
        label="Actual System",
        marker="x",
        linestyle="-",
        markersize=3,
    )
    plt.title("Comparison of RBF Model vs Actual Mass-Spring-Damper System")
    plt.xlabel("Control Input")
    plt.ylabel("Position")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Initialize the MassSpringDamper system
    msd = SpringDamper(
        mass=torch.tensor(1.0),
        damping=torch.tensor(0.5),
        spring=torch.tensor(2.0),
        dt=torch.tensor(0.1),
    )

    # Generate training data
    X, y = generate_training_data(msd)

    # Compute mean and std
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
        hidden_features=20,
    )

    losses = train_rbf_model(
        rbf_model,
        X,
        y,
        num_epochs=1000,
        batch_size=32,
        learning_rate=0.001,
    )

    # Save the trained model (including the normalizers as part of the model)
    save_load.save_rbf_model(rbf_model, "sys_rbf_springdamper.pth")

    # Plot training loss
    plot_training_loss(losses)

    # Compare predictions
    control_inputs, rbf_positions, actual_positions = compare_predictions(
        rbf_model, msd
    )

    # Plot comparison
    plot_comparison(control_inputs, rbf_positions, actual_positions)

    # Calculate and print Mean Squared Error
    mse = np.mean((np.array(rbf_positions) - np.array(actual_positions)) ** 2)
    print(f"Mean Squared Error between RBF model and actual system: {mse:.6f}")
