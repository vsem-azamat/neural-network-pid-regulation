import torch
import torch.optim as optim

from entities.pid import PID
from entities.systems.thermal import Thermal
from models.pid_lstm import LSTMAdaptivePID

from utils import save_load
from utils.plot import DynamicPlot
from utils.run import run_simulation
from classes.simulation import SimulationConfig, SimulationResults, LearningConfig


def extract_rbf_input(system: Thermal, results: SimulationResults) -> torch.Tensor:
    rbf_input = torch.tensor([
        system.X, 
        system.dXdT, 
        results.control_outputs[-1] if results.control_outputs else 0.0
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


def custom_loss(results: SimulationResults, config: SimulationConfig, step: int) -> torch.Tensor:
    left_slice = max(0, step - config.sequence_length)
    right_slice = step

    # Slices
    positions = results.rbf_predictions[left_slice:right_slice]
    setpoints = results.setpoints[left_slice:right_slice]

    # Tensors
    positions_tensor = torch.stack(positions)
    setpoints_tensor = torch.stack(setpoints)

    # Errors
    tracking_error = torch.mean((positions_tensor - setpoints_tensor) ** 2)
    overshoot = torch.mean(torch.relu(positions_tensor - setpoints_tensor))

    loss = (
        1 * tracking_error +
        1 * overshoot
    )
    return loss


if __name__ == "__main__":
    learning_config = LearningConfig(
        dt=torch.tensor(0.2),
        num_epochs=7,
        train_time=200.,
        learning_rate=0.01,
    )
    dt, num_epochs, train_steps, lr = learning_config.dt, learning_config.num_epochs, learning_config.train_steps, learning_config.learning_rate

    thermal_capacity = torch.tensor(1000.0)  # J/K
    heat_transfer_coefficient = torch.tensor(10.0)  # W/K
    initial_Kp, initial_Ki, initial_Kd = torch.tensor(100.0), torch.tensor(1.0), torch.tensor(10.0)
    input_size, hidden_size, output_size = 4, 20, 3

    thermal = Thermal(thermal_capacity, heat_transfer_coefficient, dt)
    pid = PID(initial_Kp, initial_Ki, initial_Kd)
    pid.set_limits(torch.tensor(100000), torch.tensor(0.0))  # Heat input can't be negative. [W]
    lstm_model = LSTMAdaptivePID(input_size, hidden_size, output_size)
    optimizer = optim.SGD(
        lstm_model.parameters(), 
        lr=lr,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    rbf_model = save_load.load_rbf_model('sys_rbf_thermal.pth')

    print("Training phase:")
    dynamic_plot = DynamicPlot("Thermal")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        thermal.reset()
        
        setpoints = [torch.rand(1) * 200] * train_steps  # [0, 200] degC
        training_config = SimulationConfig(
            setpoints=setpoints, 
            dt=dt,
            sequence_length=(len(setpoints)-1)//1,
            sequence_step=10,
        )
        train_results = run_simulation(
            system=thermal,
            pid=pid,
            lstm_model=lstm_model,
            rbf_model=rbf_model,
            simulation_config=training_config,
            optimizer=optimizer,
            session='train',
            extract_rbf_input=extract_rbf_input,
            extract_lstm_input=extract_lstm_input,
            loss_function=custom_loss,
        )
        dynamic_plot.update_plot(
            train_results, 
            f'Train {epoch + 1}/{num_epochs}',
            'train',
        )

        scheduler.step()
        torch.clear_autocast_cache()

    # Save the trained model
    save_load.save_model(lstm_model, 'pid_lstm_thermal.pth')

    # Validation phase
    num_validation_epochs = 1
    validation_time = 300
    validation_steps = int(validation_time / dt.item())
    print("Validation phase:")
    for epoch in range(num_validation_epochs):
        print(f"Validation Epoch {epoch + 1}/{num_validation_epochs}")
        setpoints_val = [torch.rand(1) * 200] * validation_steps # [0, 200] degC

        # >>> VALIDATION
        thermal.reset()
        pid.update_gains(initial_Kp, initial_Ki, initial_Kd)
        validation_config = SimulationConfig(
            setpoints=setpoints_val,
            sequence_length=(len(setpoints_val)-1)//20,
            sequence_step=10,
            dt=dt,
        )
        validation_results = run_simulation(
            system=thermal,
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
            f'Val {epoch + 1}/{num_validation_epochs}',
            'validation',
            )
        
        # >>> STATIC
        thermal.reset()
        pid.update_gains(
            torch.tensor(20.),
            torch.tensor(0.),
            torch.tensor(0.),
        ) # These are the optimal gains
        static_results = run_simulation(
            system=thermal,
            pid=pid,
            rbf_model=rbf_model,
            simulation_config=validation_config,
            session='static',
            extract_rbf_input=extract_rbf_input,
            extract_lstm_input=extract_lstm_input,
        )
        dynamic_plot.update_plot(
            static_results, 
            f'Static {epoch + 1}/{num_validation_epochs}',
            'static',
            )

    dynamic_plot.show()
    dynamic_plot.save("pid_lstm_thermal", learning_config)

    # Save the trained LSTM model
    save_load.save_model(lstm_model, 'pid_lstm_thermal.pth')
