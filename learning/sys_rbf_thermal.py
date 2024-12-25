import torch
import numpy as np

from models.sys_rbf import SystemRBFModel
from entities.systems.thermal import Thermal
from utils import save_load
from utils.run import train_rbf_model
from utils.plot import plot_rbf_training_results
from config import load_config


config = load_config("thermal")


def generate_training_data(thermal_system: Thermal, num_samples: int = 1000):
    X = torch.zeros((num_samples, 3))  # [temperature, temp_derivative, control_input]
    y = torch.zeros((num_samples, 1))  # next_temperature

    for i in range(num_samples):
        temperature = torch.rand(1) * 500.0
        temp_derivative = torch.rand(1) * 100.0 - 50.0  # [-50, 50]
        control_input = torch.rand(1) * 10000.0  # [0, 1000]

        thermal_system.temperature = temperature
        thermal_system.temp_derivative = temp_derivative
        next_temperature = thermal_system.apply_control(control_input)

        X[i] = torch.tensor(
            [temperature.item(), temp_derivative.item(), control_input.item()]
        )
        y[i] = next_temperature

    return X, y


def compare_predictions(model, thermal_system: Thermal, num_steps: int = 200):
    initial_temperature = torch.tensor(25.0)
    initial_temp_derivative = torch.tensor(0.0)
    control_inputs = torch.linspace(0, 1000, num_steps)

    rbf_temperatures = []
    actual_temperatures = []

    thermal_system.temperature = initial_temperature
    thermal_system.temp_derivative = initial_temp_derivative

    for control in control_inputs:
        # RBF model prediction
        with torch.no_grad():
            rbf_input = torch.tensor(
                [[thermal_system.X.item(), thermal_system.dXdT.item(), control.item()]]
            )
            rbf_next_temp = model(rbf_input).item()
            rbf_temperatures.append(rbf_next_temp)

        # Actual thermal system
        actual_next_temp = thermal_system.apply_control(control).item()
        actual_temperatures.append(actual_next_temp)

        # Update thermal system state for next iteration
        thermal_system.temperature = torch.tensor(actual_next_temp)

    return control_inputs.numpy(), rbf_temperatures, actual_temperatures


if __name__ == "__main__":
    # Initialize the Thermal system
    thermal_system = Thermal(
        thermal_capacity=torch.tensor(config.system["thermal_capacity"]),
        heat_transfer_coefficient=torch.tensor(config.system["heat_transfer_coefficient"]),
        dt=torch.tensor(config.learning.dt),
    )

    # Generate training data
    X, y = generate_training_data(thermal_system)

    # Compute mean and std
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0, unbiased=False)
    y_mean = y.mean(dim=0)
    y_std = y.std(dim=0, unbiased=False)

    # Initialize and train the RBF model
    rbf_model = SystemRBFModel(
        input_size=3,
        input_mean=X_mean,
        input_std=X_std,
        output_mean=y_mean,
        output_std=y_std,
        hidden_features=config.learning.rbf.model.hidden_size,
    )

    # Training settings
    optimizer_name = "adam"
    gradient_clip_value = None

    losses = train_rbf_model(
        rbf_model,
        X,
        y,
        num_epochs=config.learning.rbf.num_epochs,
        batch_size=config.learning.rbf.batch_size,
        learning_rate=config.learning.rbf.lr,
        optimizer=optimizer_name,
        gradient_clip_value=gradient_clip_value,
    )

    # Save the trained RBF model
    save_load.save_rbf_model(rbf_model, "sys_rbf_thermal.pth")

    # Compare RBF model predictions with actual thermal system
    control_inputs, rbf_temperatures, actual_temperatures = compare_predictions(
        rbf_model, thermal_system
    )

    # Plot comparison
    plot_rbf_training_results(
        control_inputs,
        rbf_temperatures,
        actual_temperatures,
        losses,
        system_name="Thermal",
        state_label="Temperature (C)",
        num_epochs=config.learning.rbf.num_epochs,
        learning_rate=config.learning.rbf.lr,
        optimizer_name=optimizer_name,
    )

    # Calculate the mean squared error
    mse = np.mean((np.array(rbf_temperatures) - np.array(actual_temperatures)) ** 2)
    print(f"Mean Squared Error between RBF model and actual system: {mse:.6f}")
