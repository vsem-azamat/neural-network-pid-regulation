import torch
from torch import optim

from entities.pid import PID
from entities.systems import Trolley
from models.pid_lstm import LSTMAdaptivePID

from utils import save_load
from utils.plot import DynamicPlot
from utils.run import run_simulation
from config import load_config
from learning.utils import extract_rbf_input
from classes.simulation import SimulationConfig, SimulationResults, LearningConfig


config = load_config("trolley")


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

    # Prepare LSTM input
    lstm_input = input_array.transpose(0, 1).unsqueeze(0)
    return lstm_input


def custom_loss(
    results: SimulationResults, config: SimulationConfig, step: int
) -> torch.Tensor:
    left_slice = max(0, step - config.sequence_length)
    right_slice = step

    # Slices
    # filter_ = lambda x: x[::-1][::config.sequence_step][::-1][left_slice:right_slice]
    filter_ = lambda x: x[left_slice:right_slice]
    positions = filter_(results.rbf_predictions)
    setpoints = filter_(results.setpoints)
    kp_values = filter_(results.kp_values)
    ki_values = filter_(results.ki_values)
    kd_values = filter_(results.kd_values)

    # Tensors
    positions_tensor = torch.stack(positions)
    setpoints_tensor = torch.stack(setpoints)
    kp_tensor = torch.stack(kp_values)
    ki_tensor = torch.stack(ki_values)
    kd_tensor = torch.stack(kd_values)

    # Errors
    tracking_error = torch.mean((positions_tensor - setpoints_tensor) ** 2)
    overshoot = torch.mean(torch.relu(positions_tensor - setpoints_tensor))
    kp_gain = torch.mean(kp_tensor**2)
    ki_gain = torch.mean(ki_tensor**2)
    kd_gain = torch.mean(kd_tensor**2)

    loss = (
        1 * tracking_error
        + 1 * overshoot
        # 0.1 * kp_gain +
        # 0.1 * ki_gain +
        # 0.1 * kd_gain
    )
    return loss


if __name__ == "__main__":
    learning_config = LearningConfig(
        dt=torch.tensor(config.learning.dt),
        num_epochs=config.learning.lstm.num_epochs,
        train_time=config.learning.lstm.train_time,
        learning_rate=config.learning.lstm.optimizer.lr,
    )

    mass, spring, friction = (
        torch.tensor(config.system["mass"]),
        torch.tensor(config.system["spring"]),
        torch.tensor(config.system["friction"]),
    )
    initial_Kp, initial_Ki, initial_Kd = (
        torch.tensor(10.0),
        torch.tensor(0.1),
        torch.tensor(1.0),
    )
    trolley = Trolley(
        mass=torch.tensor(config.system["mass"]),
        spring=torch.tensor(config.system["spring"]),
        friction=torch.tensor(config.system["friction"]),
        dt=torch.tensor(config.learning.dt),
    )
    pid = PID(initial_Kp, initial_Ki, initial_Kd)
    pid.set_limits(torch.tensor(50.0), torch.tensor(-50.0))
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
    rbf_model = save_load.load_rbf_model("sys_rbf_trolley.pth")

    print("Training phase:")
    dynamic_plot = DynamicPlot("Trolley")
    for epoch in range(learning_config.num_epochs):
        print(f"Epoch {epoch + 1}/{learning_config.num_epochs}")
        trolley.reset()

        setpoints = [torch.randn(1) * 10] * learning_config.train_steps
        trainining_config = SimulationConfig(
            setpoints=setpoints,
            dt=learning_config.dt,
            sequence_length=config.learning.lstm.sequence_length,
            sequence_step=config.learning.lstm.sequence_step,
            pid_gain_factor=config.learning.lstm.pid_gain_factor,
        )
        train_results = run_simulation(
            system=trolley,
            pid=pid,
            lstm_model=lstm_model,
            rbf_model=rbf_model,
            simulation_config=trainining_config,
            optimizer=optimizer,
            session="train",
            extract_rbf_input=extract_rbf_input.trolley,
            extract_lstm_input=extract_lstm_input,
            loss_function=custom_loss,
        )

        dynamic_plot.update_plot(
            train_results,
            f"Train {epoch + 1}/{learning_config.num_epochs}",
            "train",
        )

        scheduler.step()

    # Save the trained LSTM model
    save_load.save_model(lstm_model, "pid_lstm_trolley.pth")

    # Validation phase
    num_validation_epochs = 5
    validation_time = 15.0
    validation_steps = int(validation_time / learning_config.dt.item()) + 1
    print("\nValidation/Static phase")
    for epoch in range(num_validation_epochs):
        print(f"Validation/Static Epoch {epoch + 1}/{num_validation_epochs}")
        setpoints_val = [torch.randn(1) * 10] * validation_steps

        # >>> VALIDATION
        trolley.reset()
        validation_config = SimulationConfig(
            setpoints=setpoints_val,
            dt=learning_config.dt,
            sequence_length=config.learning.lstm.sequence_length,
            sequence_step=config.learning.lstm.sequence_step,
            pid_gain_factor=config.learning.lstm.pid_gain_factor,
        )
        validation_results = run_simulation(
            system=trolley,
            pid=pid,
            lstm_model=lstm_model,
            rbf_model=rbf_model,
            simulation_config=validation_config,
            session="validation",
            extract_rbf_input=extract_rbf_input.trolley,
            extract_lstm_input=extract_lstm_input,
        )
        dynamic_plot.update_plot(
            validation_results,
            f"Val {epoch + 1}/{num_validation_epochs}",
            "validation",
        )

        # >>> STATIC
        trolley.reset()
        static_results = run_simulation(
            system=trolley,
            pid=pid,
            rbf_model=rbf_model,
            simulation_config=validation_config,
            session="static",
            extract_lstm_input=extract_lstm_input,
            extract_rbf_input=extract_rbf_input.trolley,
        )
        dynamic_plot.update_plot(
            static_results,
            f"Stat {epoch + 1}/{num_validation_epochs}",
            "static",
        )

    dynamic_plot.show()
    dynamic_plot.save("pid_lstm_trolley", learning_config)

    # Save the trained LSTM model
    save_load.save_model(lstm_model, "pid_lstm_trolley.pth")
