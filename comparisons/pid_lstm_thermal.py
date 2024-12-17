import torch

from utils import save_load
from models import LSTMAdaptivePID
from entities.pid import PID
from entities.systems import Thermal
from .utils import compare_controllers_simulation, compare_controllers_metrics


if __name__ == "__main__":
    dt = torch.tensor(0.1)
    validation_time = 300
    steps = int(validation_time / dt.item()) + 1

    thermal_capacity = torch.tensor(1000.0)  # J/K
    heat_transfer_coefficient = torch.tensor(10.0)  # W/K
    thermal = Thermal(thermal_capacity, heat_transfer_coefficient, dt)
    thermal.temperature = torch.tensor(150.0) # Initial temperature [K]

    initial_Kp, initial_Ki, initial_Kd = torch.tensor(100.0), torch.tensor(1.0), torch.tensor(10.0)
    pid = PID(initial_Kp, initial_Ki, initial_Kd)
    pid.set_limits(torch.tensor(100000.0), torch.tensor(0.0))  # Heat input can't be negative. [W]

    input_size, hidden_size, output_size = 5, 20, 3
    lstm_model = LSTMAdaptivePID(input_size, hidden_size, output_size)
    lstm_model = save_load.load_model(lstm_model, "pid_lstm_thermal.pth")

    session_name = "pid_lstm_thermal"
    
    setpoints_interval = (160, 400)
    pid_gain_factor = 100

    compare_controllers_simulation(
        thermal,
        lstm_model,
        pid,
        dt,
        steps=steps,
        warm_up_steps=int(1 / dt.item()),
        random_disturbance=True,
        session_name=session_name,
        setpoints_interval=setpoints_interval,
        pid_gain_factor=pid_gain_factor
    )

    compare_controllers_metrics(
        thermal,
        lstm_model,
        pid,
        dt,
        steps=steps,
        warm_up_steps=int(1 / dt.item()),
        runs=50,
        random_disturbance=True,
        session_name=session_name,
        setpoints_interval=setpoints_interval,
        pid_gain_factor=pid_gain_factor
    )
