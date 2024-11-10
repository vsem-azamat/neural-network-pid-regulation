import torch
import numpy as np

from models.sys_rbf import SystemRBFModel
from entities.systems.trolley import Trolley
from utils import save_load
from utils.run import train_rbf_model
from utils.plot import plot_rbf_training_results


def generate_training_data(trolley: Trolley, num_samples: int = 1000):
    X = torch.zeros(
        (num_samples, 4)
    )  # [position, velocity, acceleration, control_input]
    y = torch.zeros((num_samples, 1))  # next_position

    for i in range(num_samples):
        position = torch.rand(1) * 200 - 100  # [0, 100]
        velocity = torch.rand(1) * 200 - 100  # [-100, 100]
        acceleration = torch.rand(1) * 200 - 100  # [-100, 100]
        control_input = torch.rand(1) * 200 - 100  # [-100, 100]

        trolley.position = position
        trolley.velocity = velocity
        trolley.acceleration = acceleration
        next_position = trolley.apply_control(control_input)

        X[i] = torch.tensor(
            [
                position.item(),
                velocity.item(),
                acceleration.item(),
                control_input.item(),
            ]
        )
        y[i] = next_position

    return X, y


def compare_predictions(model, trolley: Trolley, num_steps=200):
    initial_position = torch.tensor(0.5)
    initial_velocity = torch.tensor(0.0)
    initial_acceleration = torch.tensor(0.0)
    control_inputs = torch.linspace(-1, 1, num_steps)

    rbf_positions = []
    actual_positions = []

    trolley.position = initial_position
    trolley.velocity = initial_velocity
    trolley.acceleration = initial_acceleration

    for control in control_inputs:
        # RBF model prediction
        with torch.no_grad():
            rbf_input = torch.tensor(
                [
                    [
                        trolley.X.item(),
                        trolley.dXdT.item(),
                        trolley.d2XdT2.item(),
                        control.item(),
                    ]
                ]
            )
            rbf_next_position = model(rbf_input).item()
            rbf_positions.append(rbf_next_position)

        # Actual trolley system
        actual_next_position = trolley.apply_control(control).item()
        actual_positions.append(actual_next_position)

        # Update trolley state for next iteration
        trolley.position = torch.tensor(actual_next_position)

    return control_inputs.numpy(), rbf_positions, actual_positions


if __name__ == "__main__":
    # Initialize the Trolley system
    trolley = Trolley(
        mass=torch.tensor(1.0),
        spring=torch.tensor(0.5),
        friction=torch.tensor(0.1),
        dt=torch.tensor(0.1),
    )

    # Generate training data
    X, y = generate_training_data(trolley)

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

    # Training settings
    lr = 0.001
    optimizer_name = "adam"
    gradient_clip_value = None
    num_epochs = 1000
    batch_size = 64

    losses = train_rbf_model(
        rbf_model,
        X,
        y,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=lr,
        optimizer=optimizer_name,
        gradient_clip_value=gradient_clip_value,
    )  # Increased epochs

    # Save the trained model (including the normalizers as part of the model)
    save_load.save_rbf_model(rbf_model, "sys_rbf_trolley.pth")

    # Compare predictions
    control_inputs, rbf_positions, actual_positions = compare_predictions(
        rbf_model, trolley
    )

    # Plot comparison
    plot_rbf_training_results(
        control_inputs,
        rbf_positions,
        actual_positions,
        losses,
        system_name="Trolley",
        num_epochs=num_epochs,
        learning_rate=lr,
        optimizer_name=optimizer_name,
    )

    # Calculate and print Mean Squared Error
    mse = np.mean((np.array(rbf_positions) - np.array(actual_positions)) ** 2)
    print(f"Mean Squared Error between RBF model and actual system: {mse:.6f}")
