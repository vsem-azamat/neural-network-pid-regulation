import torch
from torch import optim

from entities.pid import PID
from entities.systems import Trolley
from models.pid_lstm import LSTMAdaptivePID

from utils import save_load
from utils.plot import DynamicPlot
from utils.run import run_simulation
from classes.simulation import SimulationConfig, SimulationResults, LearningConfig


def extract_rbf_input(system: Trolley, results: SimulationResults) -> torch.Tensor:
    inputs = [
        system.X,
        system.dXdT,
        system.d2XdT2,
        results.control_outputs[-1] if results.control_outputs else 0.0,
    ]
    rbf_input = torch.tensor(inputs)
    return rbf_input.unsqueeze(0)


def extract_lstm_input(
    simulation_config: SimulationConfig,
    results: SimulationResults,
) -> torch.Tensor:
    input_array = torch.zeros(4, simulation_config.sequence_length)

    # Populate input array with historical data
    error_history_len = min(
        simulation_config.sequence_length, len(results.error_history)
    )
    kp_values_len = min(simulation_config.sequence_length, len(results.kp_values))
    ki_values_len = min(simulation_config.sequence_length, len(results.ki_values))
    kd_values_len = min(simulation_config.sequence_length, len(results.kd_values))

    # Paste last values
    input_array[0, -error_history_len:] = torch.tensor(
        results.error_history[-error_history_len:]
        if results.error_history
        else [0.0] * simulation_config.sequence_length
    )
    input_array[1, -kp_values_len:] = torch.tensor(
        results.kp_values[-kp_values_len:]
        if results.kp_values
        else [0.0] * simulation_config.sequence_length
    )
    input_array[2, -ki_values_len:] = torch.tensor(
        results.ki_values[-ki_values_len:]
        if results.ki_values
        else [0.0] * simulation_config.sequence_length
    )
    input_array[3, -kd_values_len:] = torch.tensor(
        results.kd_values[-kd_values_len:]
        if results.kd_values
        else [0.0] * simulation_config.sequence_length
    )

    # Prepare LSTM input
    lstm_input = input_array.transpose(0, 1).unsqueeze(0)
    return lstm_input


def custom_loss(
    results: SimulationResults, config: SimulationConfig, step: int
) -> torch.Tensor:
    left_slice = max(0, step - config.sequence_length)
    right_slice = step

    # Slices
    positions = results.rbf_predictions[left_slice : right_slice : config.sequence_step]
    setpoints = results.setpoints[left_slice : right_slice : config.sequence_step]
    kp_values = results.kp_values[left_slice : right_slice : config.sequence_step]
    ki_values = results.ki_values[left_slice : right_slice : config.sequence_step]
    kd_values = results.kd_values[left_slice : right_slice : config.sequence_step]

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
        dt=torch.tensor(0.02),
        num_epochs=10,
        train_time=15.0,
        learning_rate=0.01,
    )
    dt, num_epochs, train_steps, lr = (
        learning_config.dt,
        learning_config.num_epochs,
        learning_config.train_steps,
        learning_config.learning_rate,
    )

    mass, spring, friction = torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.1)
    initial_Kp, initial_Ki, initial_Kd = (
        torch.tensor(10.0),
        torch.tensor(0.1),
        torch.tensor(1.0),
    )
    input_size, hidden_size, output_size = 4, 20, 3

    trolley = Trolley(mass, spring, friction, dt)
    pid = PID(initial_Kp, initial_Ki, initial_Kd)
    pid.set_limits(torch.tensor(50.0), torch.tensor(-50.0))
    lstm_model = LSTMAdaptivePID(input_size, hidden_size, output_size)
    optimizer = optim.SGD(
        lstm_model.parameters(),
        lr=lr,
        momentum=0.9,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    rbf_model = save_load.load_rbf_model("sys_rbf_trolley.pth")

    print("Training phase:")
    dynamic_plot = DynamicPlot("Trolley")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        trolley.reset()

        setpoints = [torch.randn(1) * 10] * train_steps
        trainining_config = SimulationConfig(
            setpoints=setpoints,
            dt=dt,
            sequence_length=(len(setpoints) - 1) // 1,
            sequence_step=10,
            pid_gain_factor=10,
        )
        train_results = run_simulation(
            system=trolley,
            pid=pid,
            lstm_model=lstm_model,
            rbf_model=rbf_model,
            simulation_config=trainining_config,
            optimizer=optimizer,
            session="train",
            extract_rbf_input=extract_rbf_input,
            extract_lstm_input=extract_lstm_input,
            loss_function=custom_loss,
        )

        dynamic_plot.update_plot(
            train_results,
            f"Train {epoch + 1}/{num_epochs}",
            "train",
        )

        scheduler.step()

    # Save the trained LSTM model
    save_load.save_model(lstm_model, "pid_lstm_trolley.pth")

    # Validation phase
    num_validation_epochs = 5
    validation_time = 15.0
    validation_steps = int(validation_time / dt.item())
    print("\nValidation/Static phase")
    for epoch in range(num_validation_epochs):
        print(f"Validation/Static Epoch {epoch + 1}/{num_validation_epochs}")
        setpoints_val = [torch.randn(1) * 10] * validation_steps

        # >>> VALIDATION
        trolley.reset()
        validation_config = SimulationConfig(
            setpoints=setpoints_val,
            dt=dt,
            sequence_length=50,
            pid_gain_factor=10,
        )
        validation_results = run_simulation(
            system=trolley,
            pid=pid,
            lstm_model=lstm_model,
            rbf_model=rbf_model,
            simulation_config=validation_config,
            session="validation",
            extract_rbf_input=extract_rbf_input,
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
            extract_rbf_input=extract_rbf_input,
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
