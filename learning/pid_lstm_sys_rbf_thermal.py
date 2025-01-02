import torch
import torch.optim as optim

from entities.pid import PID
from entities.systems.thermal import Thermal
from models.pid_lstm import LSTMAdaptivePID

from utils import save_load
from utils.plot import DynamicPlot
from utils.run import run_simulation
from config import load_config
from learning.utils import extract_rbf_input
from classes.simulation import SimulationConfig, SimulationResults, LearningConfig


config = load_config("thermal")


def extract_lstm_input(
    simulation_config: SimulationConfig,
    results: SimulationResults,
) -> torch.Tensor:
    input_array = torch.zeros(
        config.learning.lstm.model.input_size,
        simulation_config.sequence_length,
    )

    step = 1
    filter_ = lambda x: x[::-1][::step][::-1]
    array_positions = torch.tensor(filter_(results.positions))
    array_setpoints = torch.tensor(filter_(results.setpoints))
    array_kp_values = torch.tensor(filter_(results.kp_values))
    array_ki_values = torch.tensor(filter_(results.ki_values))
    array_kd_values = torch.tensor(filter_(results.kd_values))

    common_len = min(
        simulation_config.sequence_length,
        len(array_positions),
        len(array_setpoints),
        len(array_kp_values),
        len(array_ki_values),
        len(array_kd_values),
    )
    if common_len:
        input_array[0, -common_len:] = array_positions[-common_len::]
        input_array[1, -common_len:] = array_setpoints[-common_len::]
        input_array[2, -common_len:] = array_kp_values[-common_len::]
        input_array[3, -common_len:] = array_ki_values[-common_len::]
        input_array[4, -common_len:] = array_kd_values[-common_len::]

    lstm_input = input_array.transpose(0, 1).unsqueeze(0)
    return lstm_input


def custom_loss(
    results: SimulationResults, config: SimulationConfig, step: int
) -> torch.Tensor:
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

    loss = 1 * tracking_error + 1 * overshoot
    return loss


if __name__ == "__main__":
    learning_config = LearningConfig(
        dt=torch.tensor(config.learning.dt),
        num_epochs=config.learning.lstm.num_epochs,
        train_time=config.learning.lstm.train_time,
        learning_rate=config.learning.lstm.optimizer.lr,
    )

    thermal_capacity = torch.tensor(config.system["thermal_capacity"])  # J/K
    heat_transfer_coefficient = torch.tensor(
        config.system["heat_transfer_coefficient"]
    )  # W/K
    initial_Kp, initial_Ki, initial_Kd = (
        torch.tensor(100.0),
        torch.tensor(1.0),
        torch.tensor(10.0),
    )
    thermal = Thermal(thermal_capacity, heat_transfer_coefficient, learning_config.dt)
    pid = PID(initial_Kp, initial_Ki, initial_Kd)
    pid.set_limits(
        torch.tensor(100000), torch.tensor(0.0)
    )  # Heat input can't be negative. [W]
    lstm_model = LSTMAdaptivePID(
        input_size=config.learning.lstm.model.input_size,
        hidden_size=config.learning.lstm.model.hidden_size,
        output_size=config.learning.lstm.model.output_size,
    )
    optimizer = optim.SGD(
        lstm_model.parameters(),
        lr=config.learning.lstm.optimizer.lr,
        momentum=config.learning.lstm.optimizer.momentum,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    rbf_model = save_load.load_rbf_model("sys_rbf_thermal.pth")

    print("Training phase:")
    dynamic_plot = DynamicPlot("Thermal")
    for epoch in range(learning_config.num_epochs):
        print(f"Epoch {epoch + 1}/{learning_config.num_epochs}")
        thermal.reset()

        setpoints = [
            torch.rand(1) * 293.15 + 300
        ] * learning_config.train_steps  # [300, 593.15] K
        training_config = SimulationConfig(
            setpoints=setpoints,
            dt=learning_config.dt,
            sequence_length=config.learning.lstm.sequence_length,
            sequence_step=config.learning.lstm.sequence_step,
            pid_gain_factor=config.learning.lstm.pid_gain_factor,
        )
        train_results = run_simulation(
            system=thermal,
            pid=pid,
            lstm_model=lstm_model,
            rbf_model=rbf_model,
            simulation_config=training_config,
            optimizer=optimizer,
            session="train",
            extract_rbf_input=extract_rbf_input.thermal,
            extract_lstm_input=extract_lstm_input,
            loss_function=custom_loss,
        )
        dynamic_plot.update_plot(
            train_results,
            f"Train {epoch + 1}/{learning_config.num_epochs}",
            "train",
        )

        scheduler.step()
        torch.clear_autocast_cache()

    # Save the trained model
    save_load.save_model(lstm_model, "pid_lstm_thermal.pth")

    # Validation phase
    num_validation_epochs = 1
    validation_time = 300
    validation_steps = int(validation_time / learning_config.dt.item())
    print("Validation phase:")
    for epoch in range(num_validation_epochs):
        print(f"Validation Epoch {epoch + 1}/{num_validation_epochs}")
        setpoints_val = [
            torch.rand(1) * 293.15 + 300
        ] * validation_steps  # [300, 593.15] K

        # >>> VALIDATION
        thermal.reset()
        pid.update_gains(initial_Kp, initial_Ki, initial_Kd)
        validation_config = SimulationConfig(
            setpoints=setpoints_val,
            sequence_length=config.learning.lstm.sequence_length,
            sequence_step=config.learning.lstm.sequence_step,
            dt=learning_config.dt,
            pid_gain_factor=config.learning.lstm.pid_gain_factor,
        )
        validation_results = run_simulation(
            system=thermal,
            pid=pid,
            lstm_model=lstm_model,
            rbf_model=rbf_model,
            simulation_config=validation_config,
            session="validation",
            extract_rbf_input=extract_rbf_input.thermal,
            extract_lstm_input=extract_lstm_input,
        )
        dynamic_plot.update_plot(
            validation_results,
            f"Val {epoch + 1}/{num_validation_epochs}",
            "validation",
        )

        # >>> STATIC
        thermal.reset()
        pid.update_gains(
            torch.tensor(20.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )  # These are the optimal gains
        static_results = run_simulation(
            system=thermal,
            pid=pid,
            rbf_model=rbf_model,
            simulation_config=validation_config,
            session="static",
            extract_rbf_input=extract_rbf_input.thermal,
            extract_lstm_input=extract_lstm_input,
        )
        dynamic_plot.update_plot(
            static_results,
            f"Static {epoch + 1}/{num_validation_epochs}",
            "static",
        )

    dynamic_plot.show()
    dynamic_plot.save("pid_lstm_thermal", learning_config)

    # Save the trained LSTM model
    save_load.save_model(lstm_model, "pid_lstm_thermal.pth")
