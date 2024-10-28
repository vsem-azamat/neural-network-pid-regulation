import torch
from torch import optim

from entities.pid import PID
from entities.systems import Trolley
from models.pid_lstm import LSTMAdaptivePID

from utils import save_load
from utils.run import run_simulation, SimulationConfig, SimulationResults
from utils.plot import plot_simulation_results


def custom_loss(results: SimulationResults, config: SimulationConfig, step: int) -> torch.Tensor:
    left_slice = max(0, step - config.sequence_length)
    right_slice = step

    # Slices
    positions = results.rbf_predictions[left_slice:right_slice:config.sequence_step]
    setpoints = results.setpoints[left_slice:right_slice:config.sequence_step]
    kp_values = results.kp_values[left_slice:right_slice:config.sequence_step]
    # ki_values = results.ki_values[left_slice:right_slice:config.sequence_step]
    # kd_values = results.kd_values[left_slice:right_slice:config.sequence_step]


    # Tensors
    positions_tensor = torch.stack(positions)
    setpoints_tensor = torch.stack(setpoints)
    kp_tensor = torch.stack(kp_values)
    # ki_tensor = torch.stack(ki_values)
    # kd_tensor = torch.stack(kd_values)

    # Errors
    tracking_error = torch.mean((positions_tensor - setpoints_tensor) ** 2)
    overshoot = torch.mean(torch.relu(positions_tensor - setpoints_tensor))
    kp_gain = torch.mean(kp_tensor**2)
    # ki_gain = torch.mean(ki_tensor**2)
    # kd_gain = torch.mean(kd_tensor**2)

    loss = (
        0.5 * tracking_error +
        0.7 * overshoot
        # 0.2 * kp_gain
        # 0.1 * ki_gain + 
        # 0.1 * kd_gain
    )
    return loss


if __name__ == "__main__":
    dt = torch.tensor(0.02)
    train_time = 15.
    validation_time = 20.0
    train_steps = int(train_time / dt.item())
    validation_steps = int(validation_time / dt.item())
    num_epochs = 15

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
        # differentiable=True,
        # weight_decay=0.0001,
        # nesterov=True,
        # dampening=0.1
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    rbf_model = save_load.load_rbf_model('sys_rbf_trolley.pth')

    # Training phase
    print("Training phase:")
    epoch_results: list[SimulationResults] = []

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
            loss_function=custom_loss
        )

        epoch_results.append(train_results)
        scheduler.step()

    # Save the trained LSTM model
    save_load.save_model(lstm_model, 'pid_lstm_trolley.pth')

    # Validation phase
    print("\nValidation phase:")
    trolley.reset()
    setpoint_val = [torch.tensor(10.0)] * validation_steps
    validation_config = SimulationConfig(
        setpoints=setpoint_val,
        dt=dt,
        sequence_length=50
    )
    val_results = run_simulation(
        system=trolley,
        pid=pid,
        lstm_model=lstm_model,
        rbf_model=rbf_model,
        simulation_config=validation_config,
        session='validation',
        loss_function=custom_loss
    )

    # Plot results
    plot_simulation_results(
        training_results=epoch_results,
        training_config=SimulationConfig(setpoints, dt, sequence_length=50),

        validation_result=val_results,
        validation_config=validation_config,

        system_name='Trolley',
        save_name=f'pid_lstm_trolley_ep_{num_epochs}_lr_{lr}.png'
    )
