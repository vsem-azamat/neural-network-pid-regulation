import os
import random
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Any, Optional, List, Dict

from config import cnfg
from entities.pid import PID
from entities.systems import BaseSystem
from classes.simulation import SimulationConfig, SimulationResults
from learning.pid_lstm_sys_rbf_trolley import extract_lstm_input


def run_simulation_for_comparison(
    trolley: BaseSystem,
    pid: PID,
    lstm_regulator: Optional[torch.nn.Module],
    config: SimulationConfig[torch.Tensor],
    warm_up_steps: int,
) -> SimulationResults[torch.Tensor]:
    with torch.no_grad():
        trolley.reset()
        results = SimulationResults().with_length(config.num_steps)
        hidden = None

        for step in range(config.num_steps):
            current_time = step * config.dt
            current_position = trolley.X
            setpoint = config.setpoints[step]
            error = setpoint - current_position
            results.error_history[step] = error
            error_diff = (error - results.error_history[step - 1]) if step > 0 else torch.tensor(0.0)
            results.error_diff_history[step] = error_diff

            if lstm_regulator and step >= warm_up_steps:
                update_pid_with_lstm(lstm_regulator, pid, config, results, hidden)

            control_output = pid.compute(error, config.dt)
            trolley.apply_control(control_output)

            update_simulation_results(results, step, current_time, current_position, control_output, pid)

    return results


def update_pid_with_lstm(
    lstm_regulator: torch.nn.Module,
    pid: PID,
    config: SimulationConfig[torch.Tensor],
    results: SimulationResults[torch.Tensor],
    hidden: Optional[torch.Tensor]
) -> None:
    lstm_input = extract_lstm_input(config, results)
    pid_params, hidden = lstm_regulator(lstm_input, hidden)
    kp, ki, kd = pid_params[0] * config.pid_gain_factor
    pid.update_gains(kp.item(), ki.item(), kd.item())


def update_simulation_results(
    results: SimulationResults[torch.Tensor],
    step: int,
    current_time: torch.Tensor,
    current_position: torch.Tensor,
    control_output: torch.Tensor,
    pid: PID
) -> None:
    results.time_points[step] = current_time
    results.positions[step] = current_position
    results.control_outputs[step] = control_output
    results.kp_values[step] = pid.Kp
    results.ki_values[step] = pid.Ki
    results.kd_values[step] = pid.Kd


def plot_simulation_results(
    lstm_results: SimulationResults[torch.Tensor],
    default_results: SimulationResults[torch.Tensor],
    setpoint: float | int,
    warm_up_steps: int,
    dt: float | int,
    session_name: Optional[str] = None
) -> None:
    sns.set_context("notebook")
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["legend.fontsize"] = 12

    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    plot_position(axs[0], lstm_results, default_results, setpoint, warm_up_steps, dt)
    plot_control_output(axs[1], lstm_results, default_results, warm_up_steps, dt)
    plot_pid_parameters(axs[2], lstm_results, default_results, warm_up_steps, dt)

    plt.tight_layout()
    if session_name:
        save_plot(fig, session_name, "simulation_results.pdf")


def plot_position(
    ax: Axes,
    lstm_results: SimulationResults[torch.Tensor],
    default_results: SimulationResults[torch.Tensor],
    setpoint: float | int,
    warm_up_steps: int,
    dt: float | int
) -> None:
    ax.plot(lstm_results.time_points, lstm_results.positions, label="LSTM PID", color="#1f77b4", linewidth=2)
    ax.plot(default_results.time_points, default_results.positions, label="Default PID", color="#ff7f0e", linewidth=2)
    ax.axhline(y=setpoint, color="r", linestyle="--", label="Setpoint", linewidth=2)
    ax.axvline(x=warm_up_steps * dt, color="g", linestyle="--", label="LSTM Start", linewidth=2)
    ax.set_title("Pozice", fontweight="bold")
    ax.set_xlabel("Čas (s)")
    ax.set_ylabel("Pozice")
    ax.legend(loc="best")


def plot_control_output(
    ax: Axes,
    lstm_results: SimulationResults[torch.Tensor],
    default_results: SimulationResults[torch.Tensor],
    warm_up_steps: int,
    dt: float | int
) -> None:
    ax.plot(lstm_results.time_points, lstm_results.control_outputs, label="LSTM Control", color="#2ca02c", linewidth=2)
    ax.plot(default_results.time_points, default_results.control_outputs, label="Default Control", color="#d62728", linewidth=2)
    ax.axvline(x=warm_up_steps * dt, color="g", linestyle="--", linewidth=2)
    ax.set_title("Řídicí výstup", fontweight="bold")
    ax.set_xlabel("Čas (s)")
    ax.set_ylabel("Řídicí výstup")
    ax.legend(loc="best")


def plot_pid_parameters(
    ax: Axes,
    lstm_results: SimulationResults[torch.Tensor],
    default_results: SimulationResults[torch.Tensor],
    warm_up_steps: int,
    dt: float | int
) -> None:
    ax.plot(lstm_results.time_points, lstm_results.kp_values, label="LSTM Kp", color="#1f77b4", linewidth=2)
    ax.plot(lstm_results.time_points, lstm_results.ki_values, label="LSTM Ki", color="#ff7f0e", linewidth=2)
    ax.plot(lstm_results.time_points, lstm_results.kd_values, label="LSTM Kd", color="#2ca02c", linewidth=2)
    ax.plot(default_results.time_points, default_results.kp_values, label="Default Kp", color="#1f77b4", linestyle="--", linewidth=2)
    ax.plot(default_results.time_points, default_results.ki_values, label="Default Ki", color="#ff7f0e", linestyle="--", linewidth=2)
    ax.plot(default_results.time_points, default_results.kd_values, label="Default Kd", color="#2ca02c", linestyle="--", linewidth=2)
    ax.axvline(x=warm_up_steps * dt, color="g", linestyle="--", linewidth=2)
    ax.set_title("PID parametry", fontweight="bold")
    ax.set_xlabel("Čas (s)")
    ax.set_ylabel("Hodnota parametru")
    ax.legend(loc="upper right")


def save_plot(fig, session_name, plot_name):
    plot_path = os.path.join(cnfg.METRICS_PLOTS, session_name, plot_name)
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)


def calculate_metrics(
    positions: List[torch.Tensor],
    setpoint: torch.Tensor,
    start_index: int,
    dt: torch.Tensor
) -> Dict[str, Any]:
    actual_positions = positions[start_index:]
    times = np.arange(len(actual_positions)) * dt.item()
    error = np.array([float(p - setpoint) for p in actual_positions])

    mse = np.mean(error**2)
    settling_time = calculate_settling_time(error, setpoint.item(), dt)
    overshoot = calculate_overshoot(actual_positions, setpoint.item())
    iae = np.trapezoid(np.abs(error), times)
    ise = np.trapezoid(error**2, times)
    itae = np.trapezoid(times * np.abs(error), times)

    return {
        "MSE": mse,
        "SettlingTime": settling_time,
        "Overshoot": overshoot,
        "IAE": iae,
        "ISE": ise,
        "ITAE": itae,
    }


def calculate_settling_time(error: np.ndarray, setpoint: float, dt: torch.Tensor) -> float:
    threshold = 0.05 * abs(setpoint)
    try:
        settling_idx = next(i for i, e in enumerate(np.abs(error)) if e < threshold)
    except StopIteration:
        settling_idx = len(error) - 1
    return settling_idx * dt.item()


def calculate_overshoot(actual_positions: List[torch.Tensor], setpoint: float) -> float:
    overshoot_val = max(0, (max([p.item() for p in actual_positions]) - setpoint))
    return (overshoot_val / abs(setpoint)) * 100 if setpoint != 0 else 0.0


def plot_single_metric_comparison(
    lstm_metric_values: List[float],
    default_metric_values: List[float],
    metric_name: str,
    title_prefix: str = "",
    session_name: Optional[str] = None
) -> None:
    sns.set_context("notebook")
    sns.set_style("whitegrid")

    lstm_values = np.array(lstm_metric_values)
    default_values = np.array(default_metric_values)

    lstm_mean_value = np.mean(lstm_values)
    lstm_std_value = np.std(lstm_values)
    default_mean_value = np.mean(default_values)
    default_std_value = np.std(default_values)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"{title_prefix} {metric_name} Distribuce", fontweight="bold", y=0.98)

    plot_metric_distribution(axs[0], default_values, default_mean_value, default_std_value, "PID standardní")
    plot_metric_distribution(axs[1], lstm_values, lstm_mean_value, lstm_std_value, "PID s LSTM")

    plt.tight_layout(rect=(0., 0., 1., 0.95))
    if session_name:
        save_plot(fig, session_name, f"{metric_name}.pdf")


def plot_metric_distribution(ax, values, mean_value, std_value, title):
    sns.histplot(values, ax=ax, kde=True, color='#4c72b0', bins=10, edgecolor="black", alpha=0.8)
    ax.axvline(mean_value, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_value:.2f}')
    ax.axvline(mean_value + std_value, color='g', linestyle='--', linewidth=2, label=f'Std Dev: +{std_value:.2f}')
    ax.axvline(mean_value - std_value, color='g', linestyle='--', linewidth=2, label=f'Std Dev: -{std_value:.2f}')
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Hodnota")
    ax.set_ylabel("Počet simulací")
    ax.legend(loc="upper right")


def plot_aggregated_metrics_comparison(
    lstm_metrics_list: List[Dict[str, float]],
    default_metrics_list: List[Dict[str, float]],
    title_prefix: str = "",
    session_name: Optional[str] = None
) -> None:
    keys = lstm_metrics_list[0].keys()
    lstm_data = {k: [m[k] for m in lstm_metrics_list] for k in keys}
    default_data = {k: [m[k] for m in default_metrics_list] for k in keys}

    for metric_name in keys:
        plot_single_metric_comparison(
            lstm_data[metric_name],
            default_data[metric_name],
            metric_name,
            title_prefix,
            session_name,
        )


def compare_controllers_simulation(
    trolley: BaseSystem,
    lstm_regulator: torch.nn.Module,
    pid: PID,
    dt: torch.Tensor,
    steps: int,
    warm_up_steps: int,
    random_disturbance: bool = False,
    session_name: Optional[str] = None,
    setpoints_interval: tuple = (-20, 20),
    initial_pid_coefficients: tuple = (5.0, 0.5, 1.0),
    pid_gain_factor: int = 50
) -> None:
    disturbance = float(np.random.uniform(-0.5, 0.5)) if random_disturbance else 0.0
    setpoint_val = random.randint(*setpoints_interval)
    setpoints = [torch.tensor(setpoint_val) for _ in range(steps)] 
    config = SimulationConfig(setpoints=setpoints, dt=dt, pid_gain_factor=pid_gain_factor)

    pid.reset()
    trolley.reset()
    default_results = run_simulation_for_comparison(trolley, pid, None, config, warm_up_steps)

    initial_Kp, initial_Ki, initial_Kd = map(torch.tensor, initial_pid_coefficients)
    pid.reset()
    pid.update_gains(initial_Kp, initial_Ki, initial_Kd)
    trolley.reset()
    lstm_results = run_simulation_for_comparison(trolley, pid, lstm_regulator, config, warm_up_steps)

    plot_simulation_results(lstm_results, default_results, setpoints[-1].item(), warm_up_steps, dt.item(), session_name)


def compare_controllers_metrics(
    trolley: BaseSystem,
    lstm_regulator: torch.nn.Module,
    pid: PID,
    dt: torch.Tensor,
    steps: int,
    warm_up_steps: int,
    runs: int = 1,
    random_disturbance: bool = False,
    session_name: Optional[str] = None,
    setpoints_interval: tuple = (-20, 20),
    initial_pid_coefficients: tuple = (5.0, 0.5, 1.0),
    pid_gain_factor: int = 50
) -> None:
    lstm_metrics_all = []
    default_metrics_all = []

    for run_idx in range(runs):
        disturbance = float(np.random.uniform(-0.5, 0.5)) if random_disturbance else 0.0
        setpoint_val = random.randint(*setpoints_interval)
        setpoints = [torch.tensor(setpoint_val) for _ in range(steps)] 
        config = SimulationConfig(setpoints=setpoints, dt=dt, pid_gain_factor=pid_gain_factor)

        pid.reset()
        trolley.reset()
        default_results = run_simulation_for_comparison(trolley, pid, None, config, warm_up_steps)

        initial_Kp, initial_Ki, initial_Kd = map(torch.tensor, initial_pid_coefficients)
        pid.reset()
        pid.update_gains(initial_Kp, initial_Ki, initial_Kd)
        trolley.reset()
        lstm_results = run_simulation_for_comparison(trolley, pid, lstm_regulator, config, warm_up_steps)

        lstm_m = calculate_metrics(lstm_results.positions, setpoints[-1], warm_up_steps, dt)
        default_m = calculate_metrics(default_results.positions, setpoints[-1], warm_up_steps, dt)

        lstm_metrics_all.append(lstm_m)
        default_metrics_all.append(default_m)

    print_aggregated_stats(lstm_metrics_all, "LSTM PID", runs)
    print_aggregated_stats(default_metrics_all, "Default PID", runs)

    plot_aggregated_metrics_comparison(
        lstm_metrics_all, 
        default_metrics_all, 
        title_prefix="Porovnání PID regulátorů:",
        session_name=session_name
    )


def print_aggregated_stats(metrics_list: List[Dict[str, float]], name: str, runs: int) -> None:
    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    std_metrics = {k: np.std([m[k] for m in metrics_list]) for k in metrics_list[0]}
    print(f"\n{name} Metrics over {runs} runs:")
    for k in avg_metrics:
        print(f"{k}: Mean={avg_metrics[k]:.4f}, Std={std_metrics[k]:.4f}")
