import torch
import torch.optim as optim

from entities.pid import PID
from entities.systems import Trolley
from models.sys_rbf import SystemRBFModel
from models.pid_lstm import LSTMAdaptivePID

from .utils import plot_simulation_results, run_simulation


if __name__ == "__main__":
    dt = torch.tensor(0.01)
    train_time = 60.0
    validation_time = 30.0
    train_steps = int(train_time / dt.item())
    validation_steps = int(validation_time / dt.item())
    num_epochs = 10

    mass, spring, friction = torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.1)
    initial_Kp, initial_Ki, initial_Kd = torch.tensor(10.0), torch.tensor(0.1), torch.tensor(1.0)
    input_size, hidden_size, output_size = 3, 50, 3

    trolley = Trolley(mass, spring, friction, dt)
    pid = PID(initial_Kp, initial_Ki, initial_Kd)
    pid.set_limits(torch.tensor(50.0), torch.tensor(-50.0))
    lstm_model = LSTMAdaptivePID(input_size, hidden_size, output_size)
    optimizer = optim.SGD(lstm_model.parameters(), lr=0.002, momentum=0.2, nesterov=True)

    # Load train the RBF model and normalizers 
    from utils.save_load import load_model, load_pickle, save_model
    rbf_model = load_model(SystemRBFModel(hidden_features=20), 'sys_rbf_trolley.pth', weights_only=True)
    X_normalizer, y_normalizer = load_pickle('sys_rbf_normalizers.pkl')

    # Training phase
    print("Training phase:")
    epoch_results = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        trolley.reset()
        
        # Generate 3 random setpoints for each epoch
        setpoints = [torch.rand(1) * 10.0 + 1.0 for _ in range(3)]  # Random setpoints between 1.0 and 11.0
        print(f"Setpoints for this epoch: {[sp.item() for sp in setpoints]}")
        
        train_results = run_simulation(trolley, pid, lstm_model, rbf_model, X_normalizer, y_normalizer, setpoints, train_steps, dt, optimizer, train=True)
        epoch_results.append(train_results)

    # Save the trained LSTM model
    save_model(lstm_model, 'pid_lstm_trolley.pth')

    # Validation phase
    print("\nValidation phase:")
    trolley.reset()
    setpoint_val = torch.tensor(1.5)
    val_results = run_simulation(trolley, pid, lstm_model, rbf_model, X_normalizer, y_normalizer, [setpoint_val], validation_steps, dt, train=False, sequence_length=300)

    # Plot results

    plot_simulation_results(epoch_results, val_results, setpoints[-1], setpoint_val, system_name='Trolley', save_name='pid_lstm_sys_rbf_trolley')
    # Print final results
    final_train_results = epoch_results[-1]
    print(f"Final training position: {final_train_results[1][-1]:.4f}")
    print(f"Final training Kp: {final_train_results[3][-1]:.4f}, Ki: {final_train_results[4][-1]:.4f}, Kd: {final_train_results[5][-1]:.4f}")
    print(f"Final validation position: {val_results[1][-1]:.4f}")
    print(f"Final validation Kp: {val_results[3][-1]:.4f}, Ki: {val_results[4][-1]:.4f}, Kd: {val_results[5][-1]:.4f}")
