import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from utils.normalizer import Normalizer
from models.sys_rbf import SystemRBFModel
from entities.systems.thermal import Thermal
from utils.save_load import save_model, save_pickle

def generate_training_data(thermal_system, num_samples=2000):
    X = torch.zeros((num_samples, 2))  # [temperature, control_input]
    y = torch.zeros((num_samples, 1))  # next_temperature

    for i in range(num_samples):
        temperature = torch.rand(1) * 100  # Random temperature between 0 and 100
        control_input = torch.rand(1) * 1000  # Random control input between 0 and 1000

        thermal_system.temperature = temperature
        next_temperature = thermal_system.apply_control(control_input)

        X[i] = torch.tensor([temperature.item(), control_input.item()])
        y[i] = next_temperature

    return X, y

def train_rbf_model(model, thermal_system, num_epochs=500, batch_size=32, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X, y = generate_training_data(thermal_system)
    
    # Normalize the data
    X_normalizer = Normalizer(X)
    y_normalizer = Normalizer(y)
    
    X_normalized = X_normalizer.normalize(X)
    y_normalized = y_normalizer.normalize(y)

    losses = []

    for epoch in range(num_epochs):
        epoch_losses = []
        for i in range(0, len(X_normalized), batch_size):
            batch_X = X_normalized[i:i+batch_size]
            batch_y = y_normalized[i:i+batch_size]

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return losses, X_normalizer, y_normalizer

def plot_training_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def compare_predictions(model, thermal_system, X_normalizer, y_normalizer, num_steps=200):
    initial_temperature = torch.tensor(25.0)
    control_inputs = torch.linspace(0, 1000, num_steps)

    rbf_temperatures = []
    actual_temperatures = []

    thermal_system.temperature = initial_temperature

    for control in control_inputs:
        # RBF model prediction
        with torch.no_grad():
            rbf_input = torch.tensor([[thermal_system.temperature.item(), control.item()]])
            rbf_input_normalized = X_normalizer.normalize(rbf_input)
            rbf_next_temp_normalized = model(rbf_input_normalized)
            rbf_next_temp = y_normalizer.denormalize(rbf_next_temp_normalized).item()
            rbf_temperatures.append(rbf_next_temp)

        # Actual thermal system
        actual_next_temp = thermal_system.apply_control(control).item()
        actual_temperatures.append(actual_next_temp)

        # Update thermal system state for next iteration
        thermal_system.temperature = torch.tensor(actual_next_temp)

    return control_inputs.numpy(), rbf_temperatures, actual_temperatures

def plot_comparison(control_inputs, rbf_temperatures, actual_temperatures):
    plt.figure(figsize=(12, 6))
    plt.plot(control_inputs, rbf_temperatures, label='RBF Model', marker='o', linestyle='-', markersize=3)
    plt.plot(control_inputs, actual_temperatures, label='Actual System', marker='x', linestyle='-', markersize=3)
    plt.title('Comparison of RBF Model vs Actual Thermal System')
    plt.xlabel('Control Input (W)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Initialize the Thermal system
    thermal_system = Thermal(
        thermal_capacity=torch.tensor(1000.0),
        heat_transfer_coefficient=torch.tensor(10.0),
        dt=torch.tensor(0.1)
    )

    # Initialize and train the RBF model
    rbf_model = SystemRBFModel(input_size=2, output_size=1, hidden_features=20)
    losses, X_normalizer, y_normalizer = train_rbf_model(rbf_model, thermal_system, num_epochs=500, learning_rate=0.001)

    # Save the trained model and normalizers
    save_model(rbf_model, 'sys_rbf_thermal.pth')
    save_pickle((X_normalizer, y_normalizer), 'sys_rbf_thermal_normalizers.pkl')

    # Plot training loss
    plot_training_loss(losses)

    # Compare predictions
    control_inputs, rbf_temperatures, actual_temperatures = compare_predictions(rbf_model, thermal_system, X_normalizer, y_normalizer)

    # Plot comparison
    plot_comparison(control_inputs, rbf_temperatures, actual_temperatures)

    # Calculate and print Mean Squared Error
    mse = np.mean((np.array(rbf_temperatures) - np.array(actual_temperatures))**2)
    print(f"Mean Squared Error between RBF model and actual system: {mse:.6f}")
