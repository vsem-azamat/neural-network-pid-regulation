import torch
import torch.optim as optim

from entities.pid import PID
from entities.systems.thermal import Thermal
from models.sys_rbf import SystemRBFModel
from models.pid_lstm import LSTMAdaptivePID

from .utils import calculate_angle_2p, plot_simulation_results, custom_loss

def run_simulation(thermal_system, pid, lstm_model, rbf_model, X_normalizer, y_normalizer, setpoints, steps, dt, train=True):
    error_history = []
    rbf_predictions = []
    time_points, temperatures, control_outputs = [], [], []
    kp_values, ki_values, kd_values = [], [], []
    angle_history = []
    losses = []
    hidden = None

    current_setpoint_idx = 0
    steps_per_setpoint = steps // len(setpoints)

    for step in range(steps):
        current_time = step * dt.item()
        
        # Change setpoint at intervals
        if step % steps_per_setpoint == 0 and step > 0:
            current_setpoint_idx = (current_setpoint_idx + 1) % len(setpoints)
        
        setpoint = setpoints[current_setpoint_idx]
        current_temperature = thermal_system.get_state()
        error = setpoint - current_temperature

        # Prepare the input for the RBF model
        rbf_input = torch.tensor([current_temperature.item(), control_outputs[-1] if control_outputs else 0.0]).unsqueeze(0)
        rbf_input_normalized = X_normalizer.normalize(rbf_input)
        rbf_pred_normalized = rbf_model(rbf_input_normalized)
        rbf_pred = y_normalizer.denormalize(rbf_pred_normalized)

        # Combine the error and RBF prediction as the LSTM input
        lstm_input = torch.tensor([error.item(), rbf_pred[0].item()]).unsqueeze(0).unsqueeze(0)

        pid_params, hidden = lstm_model(lstm_input, hidden)
        kp, ki, kd = pid_params[0] * 5

        pid.update_gains(kp.item(), ki.item(), kd.item())
        control_output = pid.compute(error, dt)
        thermal_system.apply_control(control_output)

        time_points.append(current_time)
        temperatures.append(current_temperature.item())
        control_outputs.append(control_output.item())

        rbf_predictions.append(rbf_pred.item())
        error_history.append(error.item())

        kp_values.append(kp.item())
        ki_values.append(ki.item())
        kd_values.append(kd.item())

        if len(temperatures) >= 2:
            angle = calculate_angle_2p((time_points[-2], temperatures[-2]), (time_points[-1], temperatures[-1]))
            angle_history.append(angle)

        if train and step % 10 == 0 and step > 0:
            optimizer.zero_grad()
            sequence_length = min(200, len(error_history))
            input_sequence = torch.tensor([
                error_history[-sequence_length:],
                rbf_predictions[-sequence_length:]
            ]).t().unsqueeze(0)

            pid_params, _ = lstm_model(input_sequence)
            loss = custom_loss(
                torch.tensor(temperatures[-sequence_length:]),
                setpoint,
                torch.tensor(control_outputs[-sequence_length:]),
                pid_params,
                time_points[-sequence_length:]
            )
            losses.append(loss.item())
            if train:
                loss.backward()
                optimizer.step()

    return time_points, temperatures, control_outputs, kp_values, ki_values, kd_values, angle_history, losses

if __name__ == "__main__":
    dt = torch.tensor(0.1)  # Increased time step for thermal system
    train_time = 2400.0  # 1 hour of training time
    validation_time = 1800.0  # 30 minutes of validation time
    train_steps = int(train_time / dt.item())
    validation_steps = int(validation_time / dt.item())
    num_epochs = 5

    thermal_capacity = torch.tensor(1000.0)  # J/K
    heat_transfer_coefficient = torch.tensor(10.0)  # W/K
    initial_Kp, initial_Ki, initial_Kd = torch.tensor(100.0), torch.tensor(1.0), torch.tensor(10.0)
    input_size, hidden_size, output_size = 2, 20, 3

    thermal_system = Thermal(thermal_capacity, heat_transfer_coefficient, dt)
    pid = PID(initial_Kp, initial_Ki, initial_Kd)
    pid.set_limits(torch.tensor(1000.0), torch.tensor(0.0))  # Heat input can't be negative
    lstm_model = LSTMAdaptivePID(input_size, hidden_size, output_size)
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

    # Load the RBF model and normalizers 
    from utils.save_load import load_model, load_pickle, save_model
    rbf_model = load_model(SystemRBFModel(input_size=2, output_size=1, hidden_features=20), 'sys_rbf_thermal.pth')
    X_normalizer, y_normalizer = load_pickle('sys_rbf_thermal_normalizers.pkl')

    # Training phase
    print("Training phase:")
    epoch_results = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        thermal_system.reset()
        
        # Generate 3 random setpoints for each epoch
        setpoints = [torch.rand(1) * 50.0 + 20.0 for _ in range(3)]  # Random setpoints between 20°C and 70°C
        print(f"Setpoints for this epoch: {[sp.item() for sp in setpoints]}")
        
        train_results = run_simulation(thermal_system, pid, lstm_model, rbf_model, X_normalizer, y_normalizer, setpoints, train_steps, dt, train=True)
        epoch_results.append(train_results)

    # Save the trained LSTM model
    save_model(lstm_model, 'pid_lstm_thermal.pth')

    # Validation phase
    print("\nValidation phase:")
    thermal_system.reset()
    setpoint_val = torch.tensor(50.0)  # 50°C setpoint for validation
    val_results = run_simulation(thermal_system, pid, lstm_model, rbf_model, X_normalizer, y_normalizer, [setpoint_val], validation_steps, dt, train=False)

    # Plot results
    plot_simulation_results(epoch_results, val_results, setpoints[-1], setpoint_val)

    # Print final results
    final_train_results = epoch_results[-1]
    print(f"Final training temperature: {final_train_results[1][-1]:.4f}°C")
    print(f"Final training Kp: {final_train_results[3][-1]:.4f}, Ki: {final_train_results[4][-1]:.4f}, Kd: {final_train_results[5][-1]:.4f}")
    print(f"Final validation temperature: {val_results[1][-1]:.4f}°C")
    print(f"Final validation Kp: {val_results[3][-1]:.4f}, Ki: {val_results[4][-1]:.4f}, Kd: {val_results[5][-1]:.4f}")
