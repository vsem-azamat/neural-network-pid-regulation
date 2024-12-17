import torch

from utils import save_load
from models import LSTMAdaptivePID
from entities.pid import PID
from entities.systems import Trolley
from .utils import compare_controllers_simulation, compare_controllers_metrics


if __name__ == "__main__":
    dt = torch.tensor(0.02)
    validation_time = 15
    steps = int(validation_time / dt.item()) + 1

    mass, spring, friction = torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.1)
    trolley = Trolley(mass, spring, friction, dt)

    initial_Kp, initial_Ki, initial_Kd = torch.tensor(5.0), torch.tensor(0.1), torch.tensor(1.0)
    pid = PID(initial_Kp, initial_Ki, initial_Kd)
    pid.set_limits(torch.tensor(50.0), torch.tensor(-50.0))

    input_size, hidden_size, output_size = 5, 20, 3
    lstm_model = LSTMAdaptivePID(input_size, hidden_size, output_size)
    lstm_model = save_load.load_model(lstm_model, "pid_lstm_trolley.pth")

    session_name = "pid_lstm_trolley"

    setpoints_interval = (-20, 20)
    initial_pid_coefficients = (5.0, 0.5, 1.0)
    pid_gain_factor = 10
    
    compare_controllers_simulation(
        trolley,
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
        trolley,
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
