import torch

from utils import save_load
from config import load_config
from models import LSTMAdaptivePID
from entities.pid import PID
from entities.systems import Trolley
from .utils import compare_controllers_simulation, compare_controllers_metrics


config = load_config("trolley")


if __name__ == "__main__":
    dt = torch.tensor(config.learning.dt)
    validation_time = 20
    steps = int(validation_time / dt.item()) + 1

    trolley = Trolley(
        mass=torch.tensor(config.system["mass"]),
        spring=torch.tensor(config.system["spring"]),
        friction=torch.tensor(config.system["friction"]),
        dt=dt,
    )

    initial_Kp, initial_Ki, initial_Kd = (
        torch.tensor(5.0),
        torch.tensor(0.1),
        torch.tensor(1.0),
    )
    pid = PID(initial_Kp, initial_Ki, initial_Kd)
    pid.set_limits(torch.tensor(50.0), torch.tensor(-50.0))

    lstm_model = LSTMAdaptivePID(
        input_size=config.learning.lstm.model.input_size,
        hidden_size=config.learning.lstm.model.hidden_size,
        output_size=config.learning.lstm.model.output_size,
    )
    lstm_model = save_load.load_model(lstm_model, "pid_lstm_trolley.pth")

    session_name = "pid_lstm_trolley"

    setpoints_interval = (-20, 20)
    initial_pid_coefficients = (5.0, 0.5, 1.0)
    tuning_method = "pid_imc"

    compare_controllers_simulation(
        system=trolley,
        lstm_regulator=lstm_model,
        pid=pid,
        steps=steps,
        warm_up_steps=int(1 / dt.item()),
        random_disturbance=True,
        session_name=session_name,
        setpoints_interval=setpoints_interval,
        config=config,
        tuning_method=tuning_method,
    )
    compare_controllers_metrics(
        trolley,
        lstm_model,
        pid,
        steps=steps,
        warm_up_steps=int(1 / dt.item()),
        runs=100,
        random_disturbance=True,
        session_name=session_name,
        setpoints_interval=setpoints_interval,
        config=config,
        tuning_method=tuning_method,
    )
