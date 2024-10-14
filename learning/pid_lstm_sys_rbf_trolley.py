import math
import torch
from torch import optim

from entities.pid import PID
from entities.systems import Trolley
from models.pid_lstm import LSTMAdaptivePID

from utils import save_load
from utils.run import run_simulation, SimulationConfig, SimulationResults
from utils.plot import plot_simulation_results


if __name__ == "__main__":
    dt = torch.tensor(0.01)
    train_time = 30.0
    validation_time = 20.0
    train_steps = int(train_time / dt.item())
    validation_steps = int(validation_time / dt.item())
    num_epochs = 5

    mass, spring, friction = torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.1)
    initial_Kp, initial_Ki, initial_Kd = torch.tensor(10.0), torch.tensor(0.1), torch.tensor(1.0)
    input_size, hidden_size, output_size = 3, 10, 3

    trolley = Trolley(mass, spring, friction, dt)
    pid = PID(initial_Kp, initial_Ki, initial_Kd)
    pid.set_limits(torch.tensor(50.0), torch.tensor(-50.0))
    lstm_model = LSTMAdaptivePID(input_size, hidden_size, output_size)
    optimizer = optim.SGD(
        lstm_model.parameters(), 
        lr=0.002, 
        differentiable=True,
        # weight_decay=0.0001,
        # momentum=0.9,
        # nesterov=True,
        # dampening=0.1
    )

    # Load train the RBF model and normalizers 
    rbf_model = save_load.load_rbf_model('sys_rbf_trolley.pth')

    # Training phase
    print("Training phase:")
    epoch_results: list[SimulationResults] = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        trolley.reset()
        
        # setpoints = torch.zeros(train_steps)
        # intervals = 3
        # for i in range(intervals):
            # setpoints[int(i * train_steps / intervals):int((i + 1) * train_steps / intervals)] = torch.tensor(10.0) * (i + 1)
        setpoints = [torch.randn(1) * 10] * train_steps

        train_results = run_simulation(
            system=trolley,
            pid=pid,
            lstm_model=lstm_model,
            rbf_model=rbf_model,
            simulation_config=SimulationConfig(
                setpoints=setpoints, 
                dt=dt, 
                sequence_length=300,
                input_sequence_length=10,
                input_steps=10
                ),
            optimizer=optimizer,
            session='train'
        )

        epoch_results.append(train_results)

    # Save the trained LSTM model
    save_load.save_model(lstm_model, 'pid_lstm_trolley.pth')

    # Validation phase
    print("\nValidation phase:")
    trolley.reset()
    setpoint_val = [torch.tensor(10.0)] * validation_steps
    val_results = run_simulation(
        system=trolley,
        pid=pid,
        lstm_model=lstm_model,
        rbf_model=rbf_model,
        simulation_config=SimulationConfig(setpoint_val, dt, sequence_length=10),
        session='validation'
    )

    # Plot results
    plot_simulation_results(
        epoch_results=epoch_results,
        validation_result=val_results,
        system_name='Trolley',
        save_name='pid_lstm_sys_rbf_trolley'
    )
