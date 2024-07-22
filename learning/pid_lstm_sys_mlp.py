import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from entities.regulators import PID
from entities.systems.trolley import Trolley

class LSTMAdaptivePID(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMAdaptivePID, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        # Add batch dimension if it's not present
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        lstm_out, hidden = self.lstm(x, hidden)
        pid_params = self.linear(lstm_out[:, -1, :])
        return torch.exp(pid_params), hidden  # Ensure positive PID parameters

# MLP to approximate Trolley behavior
class MLPTrolley(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPTrolley, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

# Function to generate training data for MLP Trolley
def generate_trolley_data(trolley, num_samples, dt):
    data = []
    for _ in range(num_samples):
        control_input = torch.rand(1) * 2 - 1  # Random control input between -1 and 1
        current_position = trolley.get_position()
        next_position = trolley.apply_control(control_input)
        data.append((torch.tensor([current_position, control_input]), next_position))
        trolley.reset()  # Reset trolley for next sample
    return data

# Function to train MLP Trolley
def train_mlp_trolley(mlp_trolley, train_data, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp_trolley.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in train_data:
            optimizer.zero_grad()
            outputs = mlp_trolley(inputs)
            loss = criterion(outputs, targets)
            # print(f"outputs: {outputs}, targets: {targets}, loss: {loss}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_data):.4f}')

# Main execution
if __name__ == "__main__":
    # Set up the simulation parameters
    dt = torch.tensor(0.01)
    simulation_time = 10.0
    steps = int(simulation_time / dt.item())

    # Initialize the trolley and PID controller
    mass, spring, friction = torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.1)
    initial_Kp, initial_Ki, initial_Kd = torch.tensor(10.0), torch.tensor(0.1), torch.tensor(1.0)
    
    trolley = Trolley(mass, spring, friction, dt)
    pid = PID(initial_Kp, initial_Ki, initial_Kd)
    pid.set_limits(torch.tensor(50.0), torch.tensor(-50.0))

    # Initialize and train MLP Trolley
    mlp_trolley = MLPTrolley(input_size=2, hidden_size=20, output_size=1)
    train_data = generate_trolley_data(trolley, num_samples=1000, dt=dt)
    train_mlp_trolley(mlp_trolley, train_data, num_epochs=50, learning_rate=0.001)

    # Initialize LSTM Adaptive PID
    lstm_model = LSTMAdaptivePID(input_size=4, hidden_size=20, output_size=3)
    
    # Simple simulation to test the system
    setpoint = torch.tensor(1.0)
    positions = []
    mlp_positions = []
    
    for step in range(steps):
        current_position = trolley.get_position()
        mlp_input = torch.tensor([current_position, 0.0])  # Assume zero control for simplicity
        mlp_position = mlp_trolley(mlp_input)
        
        error = setpoint - current_position
        control_output = pid.compute(error, dt)
        
        # Update PID parameters using LSTM (simplified for now)
        lstm_input = torch.tensor([[error, current_position, control_output, 0.0]])  # Simplified input
        pid_params, _ = lstm_model(lstm_input)
        pid.update_gains(pid_params[0][0].item(), pid_params[0][1].item(), pid_params[0][2].item())
        
        trolley.apply_control(control_output)
        
        positions.append(current_position.item())
        mlp_positions.append(mlp_position.item())
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(positions, label='Real Trolley')
    plt.plot(mlp_positions, label='MLP Trolley')
    plt.axhline(y=setpoint.item(), color='r', linestyle='--', label='Setpoint')
    plt.xlabel('Steps')
    plt.ylabel('Position')
    plt.title('Trolley Control Simulation')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Final position (Real Trolley): {positions[-1]:.4f}")
    print(f"Final position (MLP Trolley): {mlp_positions[-1]:.4f}")