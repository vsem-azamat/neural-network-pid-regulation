import torch
from torch import optim

from entities.pid import PID
from entities.systems import Trolley
from models.pid_lstm import LSTMAdaptivePID

from utils import save_load
from utils.plot import DynamicPlot
from utils.run import run_simulation, SimulationConfig, SimulationResults


def extract_rbf_input(system: Trolley, results: SimulationResults) -> torch.Tensor:
    inputs = [
        system.X, 
        system.dXdT, 
        system.d2XdT2, 
        results.control_outputs[-1] if results.control_outputs else 0.0
    ]
    rbf_input = torch.tensor(inputs)
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
    dt = torch.tensor(0.02)
    train_time = 15.
    validation_time = 20.0
    train_steps = int(train_time / dt.item())
    validation_steps = int(validation_time / dt.item())
    num_epochs = 10

    mass, spring, friction = torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.1)
    initial_Kp, initial_Ki, initial_Kd = torch.tensor(10.0), torch.tensor(0.1), torch.tensor(1.0)
    input_size, hidden_size, output_size = 4, 20, 3

    trolley = Trolley(mass, spring, friction, dt)
    pid = PID(initial_Kp, initial_Ki, initial_Kd)
    pid.set_limits(torch.tensor(50.0), torch.tensor(-50.0))
    lstm_model = LSTMAdaptivePID(input_size, hidden_size, output_size)
    lr = 0.002
    optimizer = optim.SGD(
        lstm_model.parameters(), 
        lr=lr,
        momentum=0.9,
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    rbf_model = save_load.load_rbf_model('sys_rbf_trolley.pth')

    print("Training phase:")
    dynamic_plot = DynamicPlot('Trolley')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        trolley.reset()
        
        setpoints = [torch.randn(1) * 10] * train_steps
        trainining_config = SimulationConfig(
            setpoints=setpoints, 
            dt=dt, 
            sequence_length=(len(setpoints)-1)//2,
            sequence_step=10
        )
        train_results = run_simulation(
            system=trolley,
            pid=pid,
            lstm_model=lstm_model,
            rbf_model=rbf_model,
            simulation_config=trainining_config,
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

        scheduler.step()

    # Save the trained LSTM model
    save_load.save_model(lstm_model, 'pid_lstm_trolley.pth')

    # Validation phase
    trolley.reset()
    num_validation_epochs = 5
    print("\nValidation phase:")
    for epoch in range(num_validation_epochs):
        print(f"Validation Epoch {epoch + 1}/{num_validation_epochs}")
        setpoints_val = [torch.randn(1) * 10] * validation_steps
        validation_config = SimulationConfig(
            setpoints=setpoints_val,
            dt=dt,
            sequence_length=50
        )
        validation_results = run_simulation(
            system=trolley,
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
    save_load.save_model(lstm_model, 'pid_lstm_trolley.pth')
