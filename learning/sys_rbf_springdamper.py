import torch
import numpy as np

from models.sys_rbf import SystemRBFModel
from entities.systems.springdamper import SpringDamper
from utils import save_load
from utils.run import train_rbf_model
from utils.plot import plot_rbf_training_results


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

    # Training settings
    lr = 0.001
    optimizer_name = "adam"
    gradient_clip_value = None
    num_epochs = 1000
    batch_size = 32

    losses = train_rbf_model(
        rbf_model,
        X,
        y,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=lr,
        optimizer=optimizer_name,
        gradient_clip_value=gradient_clip_value,
    )

    # Save the trained model (including the normalizers as part of the model)
    save_load.save_rbf_model(rbf_model, "sys_rbf_springdamper.pth")

    # Compare predictions
    control_inputs, rbf_positions, actual_positions = compare_predictions(
        rbf_model, msd
    )

    # Plot comparison
    plot_rbf_training_results(
        control_inputs,
        rbf_positions,
        actual_positions,
        losses,
        system_name="SpringDamper",
    )

    # Calculate and print Mean Squared Error
    mse = np.mean((np.array(rbf_positions) - np.array(actual_positions)) ** 2)
    print(f"Mean Squared Error between RBF model and actual system: {mse:.6f}")
