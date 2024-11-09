import torch
import seaborn as sns
import matplotlib.pyplot as plt

from entities.pid import PID
from entities.systems.trolley import Trolley
from models.sys_rbf import SystemRBFModel
from models.pid_lstm import LSTMAdaptivePID
from utils.save_load import load_model, load_pickle


def compare_controllers(
    trolley,
    lstm_regulator,
    pid,
    rbf_model,
    X_normalizer,
    y_normalizer,
    setpoint,
    steps,
    dt,
    warm_up_steps,
):
    def run_simulation_for_comparison(pid, lstm_regulator=None):
        time_points, positions, control_outputs = [], [], []
        kp_values, ki_values, kd_values = [], [], []
        hidden = None
        control_output = torch.tensor(0.0)
        error_history = []

        for step in range(steps):
            current_time = step * dt.item()
            current_position = trolley.get_state()
            error = setpoint - current_position
            error_history.append(error.item())
            error_diff = (
                (error - error_history[-1]) / dt
                if len(error_history) > 0
                else torch.tensor(0.0)
            )

            if lstm_regulator and step >= warm_up_steps:
                rbf_input = torch.tensor(
                    [
                        current_position.item(),
                        trolley.velocity.item(),
                        trolley.acceleration.item(),
                        control_output.item(),
                    ]
                ).unsqueeze(0)
                rbf_input_normalized = X_normalizer.normalize(rbf_input)
                rbf_pred_normalized = rbf_model(rbf_input_normalized)
                rbf_pred = y_normalizer.denormalize(rbf_pred_normalized)
                lstm_input = (
                    torch.tensor([error.item(), error_diff.item(), rbf_pred[0].item()])
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
                pid_params, hidden = lstm_regulator(lstm_input, hidden)
                kp, ki, kd = pid_params[0] * 5
                pid.update_gains(kp.item(), ki.item(), kd.item())

            control_output = pid.compute(error, dt)
            trolley.apply_control(control_output)

            time_points.append(current_time)
            positions.append(current_position.item())
            control_outputs.append(control_output.item())
            kp_values.append(pid.Kp)
            ki_values.append(pid.Ki)
            kd_values.append(pid.Kd)

        return time_points, positions, control_outputs, kp_values, ki_values, kd_values

    # Run simulations
    trolley.reset()
    default_results = run_simulation_for_comparison(pid)
    trolley.reset()
    lstm_results = run_simulation_for_comparison(pid, lstm_regulator)

    # Set up the plot style
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["legend.fontsize"] = 12

    # Create the plot
    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    # Plot positions
    axs[0].plot(
        lstm_results[0], lstm_results[1], label="LSTM PID", color="#1f77b4", linewidth=2
    )
    axs[0].plot(
        default_results[0],
        default_results[1],
        label="Default PID",
        color="#ff7f0e",
        linewidth=2,
    )
    axs[0].axhline(y=setpoint, color="r", linestyle="--", label="Setpoint", linewidth=2)
    axs[0].axvline(
        x=warm_up_steps * dt.item(),
        color="g",
        linestyle="--",
        label="LSTM Start",
        linewidth=2,
    )
    axs[0].set_title("Position", fontweight="bold")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Position")
    axs[0].legend(loc="best")

    # Plot control outputs
    axs[1].plot(
        lstm_results[0],
        lstm_results[2],
        label="LSTM Control",
        color="#2ca02c",
        linewidth=2,
    )
    axs[1].plot(
        default_results[0],
        default_results[2],
        label="Default Control",
        color="#d62728",
        linewidth=2,
    )
    axs[1].axvline(
        x=warm_up_steps * dt.item(),
        color="g",
        linestyle="--",
        label="LSTM Start",
        linewidth=2,
    )
    axs[1].set_title("Control Output", fontweight="bold")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Control Output")
    axs[1].legend(loc="best")

    # Plot all K parameters
    axs[2].plot(
        lstm_results[0], lstm_results[3], label="LSTM Kp", color="#1f77b4", linewidth=2
    )
    axs[2].plot(
        lstm_results[0], lstm_results[4], label="LSTM Ki", color="#ff7f0e", linewidth=2
    )
    axs[2].plot(
        lstm_results[0], lstm_results[5], label="LSTM Kd", color="#2ca02c", linewidth=2
    )
    axs[2].plot(
        default_results[0],
        default_results[3],
        label="Default Kp",
        color="#1f77b4",
        linestyle="--",
        linewidth=2,
    )
    axs[2].plot(
        default_results[0],
        default_results[4],
        label="Default Ki",
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
    )
    axs[2].plot(
        default_results[0],
        default_results[5],
        label="Default Kd",
        color="#2ca02c",
        linestyle="--",
        linewidth=2,
    )
    axs[2].axvline(
        x=warm_up_steps * dt.item(),
        color="g",
        linestyle="--",
        label="LSTM Start",
        linewidth=2,
    )
    axs[2].set_title("PID Parameters", fontweight="bold")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Parameter Value")
    axs[2].legend(loc="center right")

    plt.tight_layout()
    plt.show()

    # Calculate and print performance metrics
    def calculate_metrics(positions, setpoint, start_index):
        error = [abs(p - setpoint) for p in positions[start_index:]]
        mse = sum([e**2 for e in error]) / len(error)
        settling_time = (
            next(
                (i for i, e in enumerate(error) if e < 0.05 * abs(setpoint)), len(error)
            )
            * dt.item()
        )
        overshoot = (
            max(0, max(positions[start_index:]) - setpoint) / abs(setpoint) * 100
        )
        return mse, settling_time, overshoot

    lstm_metrics = calculate_metrics(lstm_results[1], setpoint, warm_up_steps)
    default_metrics = calculate_metrics(default_results[1], setpoint, warm_up_steps)

    print("\nLSTM PID Metrics:")
    print(f"MSE: {lstm_metrics[0]:.4f}")
    print(f"Settling Time: {lstm_metrics[1]:.4f} s")
    print(f"Overshoot: {lstm_metrics[2]:.2f}%")

    print("\nDefault PID Metrics:")
    print(f"MSE: {default_metrics[0]:.4f}")
    print(f"Settling Time: {default_metrics[1]:.4f} s")
    print(f"Overshoot: {default_metrics[2]:.2f}%")


if __name__ == "__main__":
    dt = torch.tensor(0.01)
    simulation_time = 30.0
    steps = int(simulation_time / dt.item())
    warm_up_steps = int(1 / dt.item())  # 5 seconds warm-up period

    mass, spring, friction = torch.tensor(1.0), torch.tensor(0.5), torch.tensor(0.1)
    initial_Kp, initial_Ki, initial_Kd = (
        torch.tensor(1.314),
        torch.tensor(0.5),
        torch.tensor(0.707),
    )
    input_size, hidden_size, output_size = 3, 50, 3

    # Initialize the trolley, LSTM PID, and RBF model
    trolley = Trolley(mass, spring, friction, dt)
    lstm_model = LSTMAdaptivePID(input_size, hidden_size, output_size)
    lstm_model = load_model(lstm_model, "pid_lstm_trolley.pth")
    lstm_model.eval()

    rbf_model = SystemRBFModel(hidden_features=20)
    rbf_model = load_model(rbf_model, "sys_rbf_trolley.pth")
    X_normalizer, y_normalizer = load_pickle("sys_rbf_normalizers.pkl")

    print("\nComparing LSTM PID with Default PID:")
    pid = PID(initial_Kp, initial_Ki, initial_Kd)
    pid.set_limits(torch.tensor(50.0), torch.tensor(-50.0))
    compare_setpoint = torch.tensor(5.0)  # You can change this to any desired setpoint

    compare_controllers(
        trolley,
        lstm_model,
        pid,
        rbf_model,
        X_normalizer,
        y_normalizer,
        compare_setpoint,
        steps,
        dt,
        warm_up_steps,
    )
