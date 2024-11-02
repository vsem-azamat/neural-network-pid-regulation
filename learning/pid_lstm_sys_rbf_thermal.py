import torch
import torch.optim as optim

from entities.pid import PID
from entities.systems.thermal import Thermal
from models.pid_lstm import LSTMAdaptivePID

from utils import save_load
from utils.plot import DynamicPlot
from utils.run import run_simulation, SimulationConfig, SimulationResults


def extract_rbf_input(system: Thermal, results: SimulationResults) -> torch.Tensor:
    rbf_input = torch.tensor([
        system.X, 
        system.dXdT, 
        # system.d2XdT2, 
        # results.control_outputs[-1] if results.control_outputs else 0.0
    ])
    return rbf_input.unsqueeze(0)


def extract_lstm_input(
    simulation_config: SimulationConfig,
    results: SimulationResults,
) -> torch.Tensor:
    input_array = torch.zeros(4, simulation_config.sequence_length)

    # Populate input array with historical data
    error_history_len = min(simulation_config.sequence_length, len(results.error_history))
    kp_values_len = min(simulation_config.sequence_length, len(results.kp_values))
    ki_values_len = min(simulation_config.sequence_length, len(results.ki_values))
    kd_values_len = min(simulation_config.sequence_length, len(results.kd_values))

    # Paste last values
    input_array[0, -error_history_len:] = torch.tensor(
        results.error_history[-error_history_len:] if results.error_history else [0.0] * simulation_config.sequence_length
    )
    input_array[1, -kp_values_len:] = torch.tensor(
        results.kp_values[-kp_values_len:] if results.kp_values else [0.0] * simulation_config.sequence_length
    )
    input_array[2, -ki_values_len:] = torch.tensor(
        results.ki_values[-ki_values_len:] if results.ki_values else [0.0] * simulation_config.sequence_length
    )
    input_array[3, -kd_values_len:] = torch.tensor(
        results.kd_values[-kd_values_len:] if results.kd_values else [0.0] * simulation_config.sequence_length
    )

    # Prepare LSTM input
    lstm_input = input_array.transpose(0, 1).unsqueeze(0)
    return lstm_input




if __name__ == "__main__":
    dt = torch.tensor(0.2)  # Increased time step for thermal system
    train_time = 300.  # 1 hour of training time
    validation_time = 300
    train_steps = int(train_time / dt.item())
    validation_steps = int(validation_time / dt.item())
    num_epochs = 5

    thermal_capacity = torch.tensor(1000.0)  # J/K
    heat_transfer_coefficient = torch.tensor(10.0)  # W/K
    initial_Kp, initial_Ki, initial_Kd = torch.tensor(100.0), torch.tensor(1.0), torch.tensor(10.0)
    input_size, hidden_size, output_size = 4, 20, 3

    thermal_system = Thermal(thermal_capacity, heat_transfer_coefficient, dt)
    pid = PID(initial_Kp, initial_Ki, initial_Kd)
    pid.set_limits(torch.tensor(1000.0), torch.tensor(0.0))  # Heat input can't be negative
    lstm_model = LSTMAdaptivePID(input_size, hidden_size, output_size)
    lr = 0.001
    optimizer = optim.SGD(
        lstm_model.parameters(), 
        lr=lr,
    )
    rbf_model = save_load.load_rbf_model('sys_rbf_thermal.pth')

    print("Training phase:")
    dynamic_plot = DynamicPlot("Thermal")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        thermal_system.reset()
        
        setpoints = [torch.randn(1) * 100] * train_steps
        training_config = SimulationConfig(
            setpoints=setpoints, 
            dt=dt,
            sequence_length=(len(setpoints)-1)//1,
            sequence_step=10,
        )

        train_results = run_simulation(
            system=thermal_system,
            pid=pid,
            lstm_model=lstm_model,
            rbf_model=rbf_model,
            simulation_config=training_config,
            optimizer=optimizer,
            session='train',
            extract_rbf_input=extract_rbf_input,
            extract_lstm_input=extract_lstm_input,
        )

        dynamic_plot.update_plot(
            train_results, 
            f'Epoch {epoch + 1}/{num_epochs}',
            'train',
        )

    # Save the trained model
    save_load.save_model(lstm_model, 'pid_lstm_thermal.pth')

    # Validation phase
    thermal_system.reset()
    num_validation_epochs = 5
    print("Validation phase:")
    for epoch in range(num_validation_epochs):
        print(f"Validation Epoch {epoch + 1}/{num_validation_epochs}")
        setpoints_val = [torch.randn(1) * 10] * validation_steps
        validation_config = SimulationConfig(
            setpoints=setpoints_val,
            sequence_length=(len(setpoints_val)-1)//20,
            sequence_step=10,
            dt=dt,
        )
        validation_results = run_simulation(
            system=thermal_system,
            pid=pid,
            lstm_model=lstm_model,
            rbf_model=rbf_model,
            simulation_config=validation_config,
            session='validation',
            extract_rbf_input=extract_rbf_input,
            extract_lstm_input=extract_lstm_input,
        )
        dynamic_plot.update_plot(
            validation_results, 
            f'Validation Epoch {epoch + 1}/{num_validation_epochs}',
            'validation',
            )

    dynamic_plot.show()

    # Save the trained LSTM model
    save_load.save_model(lstm_model, 'pid_lstm_thermal.pth')

 