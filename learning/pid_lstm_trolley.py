import torch
import torch.optim as optim

from entities.pid import PID
from models.pid_lstm import LSTMAdaptivePID
from entities.systems.trolley import Trolley
from .utils import calculate_angle_2p, custom_loss, plot_simulation_results


def run_simulation(trolley, pid, lstm_model, setpoint, steps, dt, train=True):
    time_points, positions, control_outputs = [], [], []
    kp_values, ki_values, kd_values = [], [], []
    angle_history = []
    losses = []
    hidden = None

    for step in range(steps):
        current_time = step * dt.item()
        current_position = trolley.get_state()
        error = setpoint - current_position
        
        lstm_input = torch.tensor([error.item()]).unsqueeze(0).unsqueeze(0)

        pid_params, hidden = lstm_model(lstm_input, hidden)
        kp, ki, kd = pid_params[0] * 5
        
        pid.update_gains(kp.item(), ki.item(), kd.item())
        control_output = pid.compute(error, dt)
        trolley.apply_control(control_output)
        
        time_points.append(current_time)
        positions.append(current_position.item())
        control_outputs.append(control_output.item())
        kp_values.append(kp.item())
        ki_values.append(ki.item())
        kd_values.append(kd.item())
        
        if len(positions) >= 2:
            angle = calculate_angle_2p((time_points[-2], positions[-2]), (time_points[-1], positions[-1]))
            angle_history.append(angle)
        
        if train and step % 10 == 0 and step > 0:
            optimizer.zero_grad()
            sequence_length = min(100, len(positions))
            input_sequence = torch.tensor([positions[-sequence_length:]]).t().unsqueeze(0)
            
            pid_params, _ = lstm_model(input_sequence)
            loss = custom_loss(
                torch.tensor(positions[-sequence_length:]),
                setpoint,
                torch.tensor(control_outputs[-sequence_length:]),
                pid_params,
                time_points[-sequence_length:]
            )
            losses.append(loss.item())
            if train:
                loss.backward()
                optimizer.step()

    return time_points, positions, control_outputs, kp_values, ki_values, kd_values, angle_history, losses


if __name__ == "__main__":
    # Set up the simulation parameters
    dt = torch.tensor(0.01)
    train_time = 60.0
    validation_time = 30.0
    train_steps = int(train_time / dt.item())
    validation_steps = int(validation_time / dt.item())
    num_epochs = 5

    # Initialize the trolley, PID controller, and LSTM model
    mass, spring, friction = torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.1)
    initial_Kp, initial_Ki, initial_Kd = torch.tensor(10.0), torch.tensor(0.1), torch.tensor(1.0)
    input_size, hidden_size, output_size = 1, 20, 3

    trolley = Trolley(mass, spring, friction, dt)
    pid = PID(initial_Kp, initial_Ki, initial_Kd)
    pid.set_limits(torch.tensor(50.0), torch.tensor(-50.0))
    lstm_model = LSTMAdaptivePID(input_size, hidden_size, output_size)
    optimizer = optim.SGD(lstm_model.parameters(), lr=0.0005, momentum=0.9)

    # Training phase
    print("Training phase:")
    setpoint_train = torch.rand(1) * 5.0 + 1.0  # Random setpoint between 1.0 and 6.0
    epoch_results = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        trolley.reset()
        train_results = run_simulation(trolley, pid, lstm_model, setpoint_train, train_steps, dt, train=True)
        epoch_results.append(train_results)

    # Save the trained LSTM model
    torch.save(lstm_model.state_dict(), "pid_lstm_trolley.pth")

    # Validation phase
    print("\nValidation phase:")
    trolley.reset()  # Reset trolley to initial state
    setpoint_val = torch.tensor(1.5)  # Different setpoint for validation
    val_results = run_simulation(trolley, pid, lstm_model, setpoint_val, validation_steps, dt, train=False)

    # Plot results
    plot_simulation_results(epoch_results, val_results, setpoint_train, setpoint_val, system_name='Trolley')

    # Print final results
    final_train_results = epoch_results[-1]
    print(f"Final training position: {final_train_results[1][-1]:.4f}")
    print(f"Final training Kp: {final_train_results[3][-1]:.4f}, Ki: {final_train_results[4][-1]:.4f}, Kd: {final_train_results[5][-1]:.4f}")
    print(f"Final validation position: {val_results[1][-1]:.4f}")
    print(f"Final validation Kp: {val_results[3][-1]:.4f}, Ki: {val_results[4][-1]:.4f}, Kd: {val_results[5][-1]:.4f}")
