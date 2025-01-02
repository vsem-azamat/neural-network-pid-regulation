import os
from turtle import title
import torch
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Any, Optional, List, Dict, Literal

from config import cnfg
from config.models import ConfigPack
from entities.pid import PID
from entities.systems import BaseSystem
from classes.simulation import SimulationConfig, SimulationResults
from learning.pid_lstm_sys_rbf_trolley import extract_lstm_input


def run_simulation_for_comparison(
    system: BaseSystem,
    pid: PID,
    lstm_regulator: Optional[torch.nn.Module],
    config: ConfigPack,
    warm_up_steps: int,
    setpoints: List[torch.Tensor],
) -> SimulationResults[torch.Tensor]:
    with torch.no_grad():
        dt = torch.tensor(config.learning.dt)
        system.reset()
        results = SimulationResults()
        hidden = None

        for step in range(len(setpoints)):
            current_time = step * dt
            current_position = system.X
            setpoint = setpoints[step]
            error = setpoint - current_position

            if lstm_regulator and step >= warm_up_steps:
                update_pid_with_lstm(lstm_regulator, pid, config, results, hidden)

            control_output = pid.compute(error, dt)
            system.apply_control(control_output)

            update_simulation_results(
                results, current_time, current_position, setpoint, control_output, pid
            )

    return results


def update_pid_with_lstm(
    lstm_regulator: torch.nn.Module,
    pid: PID,
    config: ConfigPack,
    results: SimulationResults[torch.Tensor],
    hidden: Optional[torch.Tensor],
) -> None:
    config_simulation = SimulationConfig(
        setpoints=[],
        sequence_length=config.learning.lstm.sequence_length,
        dt=torch.tensor(config.learning.dt),
    )
    lstm_input = extract_lstm_input(config_simulation, results)
    pid_params, hidden = lstm_regulator(lstm_input, hidden)
    kp, ki, kd = pid_params[0] * config.learning.lstm.pid_gain_factor
    pid.update_gains(kp.item(), ki.item(), kd.item())


def update_simulation_results(
    results: SimulationResults[torch.Tensor],
    current_time: torch.Tensor,
    current_position: torch.Tensor,
    setpoint: torch.Tensor,
    control_output: torch.Tensor,
    pid: PID,
) -> None:
    error = setpoint - current_position
    error_diff = (
        (error - results.error_history[-1])
        if results.error_history
        else torch.tensor(0.0)
    )

    results.setpoints.append(setpoint)
    results.error_history.append(error)
    results.error_diff_history.append(error_diff)
    results.time_points.append(current_time)
    results.positions.append(current_position)
    results.control_outputs.append(control_output)
    results.kp_values.append(pid.Kp)
    results.ki_values.append(pid.Ki)
    results.kd_values.append(pid.Kd)


def plot_simulation_results(
    lstm_dynamic_results: SimulationResults[torch.Tensor],
    lstm_static_results: SimulationResults[torch.Tensor],
    default_results: SimulationResults[torch.Tensor],
    setpoint: float | int,
    warm_up_steps: int,
    dt: float | int,
    session_name: Optional[str] = None,
    tuning_method: str = "",
) -> None:
    sns.set_context("notebook")
    sns.set_style("whitegrid")
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["legend.fontsize"] = 12

    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

    plot_position(
        axs[0],
        lstm_dynamic_results,
        lstm_static_results,
        default_results,
        setpoint,
        warm_up_steps,
        dt,
    )
    plot_control_output(
        axs[1],
        lstm_dynamic_results,
        lstm_static_results,
        default_results,
        warm_up_steps,
        dt,
    )
    plot_pid_parameters(
        axs[2],
        lstm_dynamic_results,
        lstm_static_results,
        default_results,
        warm_up_steps,
        dt,
    )

    title = f"Výsledky simulace"
    if tuning_method:
        title += f" metoda: {tuning_method}"
    fig.suptitle(title, fontweight="bold", y=0.98)
    plt.tight_layout()
    if session_name:
        save_plot(fig, session_name, "simulation_results.pdf")


def plot_position(
    ax: Axes,
    lstm_results: SimulationResults[torch.Tensor],
    lstm_static_results: SimulationResults[torch.Tensor],
    default_results: SimulationResults[torch.Tensor],
    setpoint: float | int,
    warm_up_steps: int,
    dt: float | int,
) -> None:
    ax.plot(
        lstm_results.time_points,
        lstm_results.positions,
        label="Dynamic LSTM PID",
        color="#1f77b4",
        linewidth=2,
    )
    ax.plot(
        lstm_static_results.time_points,
        lstm_static_results.positions,
        label="Static LSTM PID",
        color="#9467bd",
        linewidth=2,
    )
    ax.plot(
        default_results.time_points,
        default_results.positions,
        label="Default PID",
        color="#ff7f0e",
        linewidth=2,
    )
    ax.axhline(y=setpoint, color="r", linestyle="--", label="Setpoint", linewidth=2)
    ax.axvline(
        x=warm_up_steps * dt, color="g", linestyle="--", label="LSTM Start", linewidth=2
    )
    ax.set_title("Pozice", fontweight="bold")
    ax.set_xlabel("Čas (s)")
    ax.set_ylabel("Pozice")
    ax.legend(loc="best")


def plot_control_output(
    ax: Axes,
    lstm_results: SimulationResults[torch.Tensor],
    lstm_static_results: SimulationResults[torch.Tensor],
    default_results: SimulationResults[torch.Tensor],
    warm_up_steps: int,
    dt: float | int,
) -> None:
    ax.plot(
        lstm_results.time_points,
        lstm_results.control_outputs,
        label="Dynamic LSTM Control",
        color="#2ca02c",
        linewidth=2,
    )
    ax.plot(
        lstm_static_results.time_points,
        lstm_static_results.control_outputs,
        label="Static LSTM Control",
        color="#9467bd",
        linewidth=2,
    )
    ax.plot(
        default_results.time_points,
        default_results.control_outputs,
        label="Default Control",
        color="#d62728",
        linewidth=2,
    )
    ax.axvline(x=warm_up_steps * dt, color="g", linestyle="--", linewidth=2)
    ax.set_title("Řídicí výstup", fontweight="bold")
    ax.set_xlabel("Čas (s)")
    ax.set_ylabel("Řídicí výstup")
    ax.legend(loc="best")


def plot_pid_parameters(
    ax: Axes,
    lstm_results: SimulationResults[torch.Tensor],
    lstm_static_results: SimulationResults[torch.Tensor],
    default_results: SimulationResults[torch.Tensor],
    warm_up_steps: int,
    dt: float | int,
) -> None:
    # Dynamic LSTM
    ax.plot(
        lstm_results.time_points,
        lstm_results.kp_values,
        label="Dynamic LSTM Kp",
        color="#1f77b4",
        linewidth=2,
    )
    ax.plot(
        lstm_results.time_points,
        lstm_results.ki_values,
        label="Dynamic LSTM Ki",
        color="#ff7f0e",
        linewidth=2,
    )
    ax.plot(
        lstm_results.time_points,
        lstm_results.kd_values,
        label="Dynamic LSTM Kd",
        color="#2ca02c",
        linewidth=2,
    )

    # Static LSTM
    ax.plot(
        lstm_static_results.time_points,
        lstm_static_results.kp_values,
        label="Static LSTM Kp",
        color="#9467bd",
        linewidth=2,
    )
    ax.plot(
        lstm_static_results.time_points,
        lstm_static_results.ki_values,
        label="Static LSTM Ki",
        color="#8c564b",
        linewidth=2,
    )
    ax.plot(
        lstm_static_results.time_points,
        lstm_static_results.kd_values,
        label="Static LSTM Kd",
        color="#e377c2",
        linewidth=2,
    )

    # Default PID
    ax.plot(
        default_results.time_points,
        default_results.kp_values,
        label="Default Kp",
        color="#1f77b4",
        linestyle="--",
        linewidth=2,
    )
    ax.plot(
        default_results.time_points,
        default_results.ki_values,
        label="Default Ki",
        color="#ff7f0e",
        linestyle="--",
        linewidth=2,
    )
    ax.plot(
        default_results.time_points,
        default_results.kd_values,
        label="Default Kd",
        color="#2ca02c",
        linestyle="--",
        linewidth=2,
    )
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
    dt: torch.Tensor,
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
    rise_time = calculate_rise_time(actual_positions, setpoint.item(), dt)

    return {
        "MSE": mse,
        "Doba ustálení (ts)": settling_time,
        "Překmit (Mp)": overshoot,
        "IAE": iae,
        "ISE": ise,
        "ITAE": itae,
        "Doba náběhu (tr)": rise_time,
    }


def calculate_settling_time(
    error: np.ndarray, setpoint: float, dt: torch.Tensor
) -> float:
    threshold = 0.02 * abs(setpoint)
    try:
        settling_idx = next(i for i, e in enumerate(np.abs(error)) if e < threshold)
    except StopIteration:
        settling_idx = len(error) - 1
    return settling_idx * dt.item()


def calculate_rise_time(
    positions: List[torch.Tensor], setpoint: float, dt: torch.Tensor
) -> float:
    threshold_90 = 0.9 * abs(setpoint)
    try:
        rise_idx = next(i for i, p in enumerate(positions) if p >= threshold_90)
    except StopIteration:
        rise_idx = len(positions) - 1
    return rise_idx * dt.item()


def calculate_overshoot(actual_positions: List[torch.Tensor], setpoint: float) -> float:
    overshoot_val = max(0, (max([p.item() for p in actual_positions]) - setpoint))
    return (overshoot_val / abs(setpoint)) * 100 if setpoint != 0 else 0.0


def plot_single_metric_comparison(
    default_metric_values: List[float],
    lstm_dynamic_values: List[float],
    lstm_static_values: List[float],
    metric_name: str,
    title_prefix: str = "",
    session_name: Optional[str] = None,
) -> None:
    sns.set_context("notebook")
    sns.set_style("whitegrid")

    lstm_values = np.array(lstm_dynamic_values)
    default_values = np.array(default_metric_values)

    lstm_mean_value = np.mean(lstm_values)
    lstm_std_value = np.std(lstm_values)
    default_mean_value = np.mean(default_values)
    default_std_value = np.std(default_values)

    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle(f"{title_prefix} {metric_name} Distribuce", fontweight="bold", y=0.98)

    plot_metric_distribution(
        axs[0], default_values, default_mean_value, default_std_value, "PID standardní"
    )
    plot_metric_distribution(
        axs[1], lstm_values, lstm_mean_value, lstm_std_value, "PID LSTM dynamický"
    )
    plot_metric_distribution(
        axs[2], lstm_static_values, lstm_mean_value, lstm_std_value, "PID LSTM statický"
    )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    if session_name:
        save_plot(fig, session_name, f"{metric_name}.pdf")


def plot_metric_distribution(ax, values, mean_value, std_value, title):
    sns.histplot(
        values, ax=ax, kde=True, color="#4c72b0", bins=10, edgecolor="black", alpha=0.8
    )
    ax.axvline(
        mean_value,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_value:.2f}",
    )
    ax.axvline(
        mean_value + std_value,
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Std Dev: +{std_value:.2f}",
    )
    ax.axvline(
        mean_value - std_value,
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Std Dev: -{std_value:.2f}",
    )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Hodnota")
    ax.set_ylabel("Počet simulací")
    ax.legend(loc="upper right")


def plot_aggregated_metrics_comparison(
    default_metrics: List[Dict[str, float]],
    lstm_dynamic_metrics: List[Dict[str, float]],
    lstm_static_metrics: List[Dict[str, float]],
    title_prefix: str = "",
    session_name: Optional[str] = None,
) -> None:
    keys = lstm_dynamic_metrics[0].keys()
    get_values = lambda metrics: {k: [m[k] for m in metrics] for k in keys}
    default_values = get_values(default_metrics)
    lstm_dynamic_values = get_values(lstm_dynamic_metrics)
    lstm_static_data = get_values(lstm_static_metrics)

    for metric_name in keys:
        plot_single_metric_comparison(
            default_metric_values=default_values[metric_name],
            lstm_dynamic_values=lstm_dynamic_values[metric_name],
            lstm_static_values=lstm_static_data[metric_name],
            metric_name=metric_name,
            title_prefix=title_prefix,
            session_name=session_name,
        )


def compare_controllers_simulation(
    system: BaseSystem,
    lstm_regulator: torch.nn.Module,
    pid: PID,
    steps: int,
    warm_up_steps: int,
    config: ConfigPack,
    random_disturbance: bool = False,
    session_name: Optional[str] = None,
    setpoints_interval: tuple = (-20, 20),
    tuning_method: Literal[
        "ziegler_nichols", "cohen_coon", "pid_imc"
    ] = "ziegler_nichols",
) -> None:
    disturbance = float(np.random.uniform(-0.5, 0.5)) if random_disturbance else 0.0
    setpoint_val = random.randint(*setpoints_interval)
    setpoints = [torch.tensor(setpoint_val) for _ in range(steps)]
    dt = config.learning.dt
    Kp_, Ki_, Kd_ = system.tune_pid(dt=dt, steps=steps, method=tuning_method)
    Kp, Ki, Kd = map(torch.tensor, (Kp_, Ki_, Kd_))
    pid.update_gains(Kp, Ki, Kd)

    pid.reset()
    system.reset()
    default_results = run_simulation_for_comparison(
        system=system,
        pid=pid,
        lstm_regulator=None,
        config=config,
        warm_up_steps=warm_up_steps,
        setpoints=setpoints,
    )

    pid.reset()
    system.reset()
    lstm_dynamic_results = run_simulation_for_comparison(
        system=system,
        pid=pid,
        lstm_regulator=lstm_regulator,
        config=config,
        warm_up_steps=warm_up_steps,
        setpoints=setpoints,
    )

    lstm_static_results = run_simulation_for_comparison(
        system=system,
        pid=pid,
        lstm_regulator=None,
        config=config,
        warm_up_steps=warm_up_steps,
        setpoints=setpoints,
    )

    plot_simulation_results(
        lstm_dynamic_results=lstm_dynamic_results,
        lstm_static_results=lstm_static_results,
        default_results=default_results,
        setpoint=setpoints[-1].item(),
        warm_up_steps=warm_up_steps,
        dt=dt,
        session_name=session_name,
        tuning_method=tuning_method,
    )


def compare_controllers_metrics(
    system: BaseSystem,
    lstm_regulator: torch.nn.Module,
    pid: PID,
    steps: int,
    warm_up_steps: int,
    config: ConfigPack,
    runs: int = 1,
    random_disturbance: bool = False,
    session_name: Optional[str] = None,
    setpoints_interval: tuple = (-20, 20),
    tuning_method: Literal[
        "ziegler_nichols", "cohen_coon", "pid_imc"
    ] = "ziegler_nichols",
) -> None:
    default_metrics = []
    lstm_dynamic_metrics = []
    lstm_static_metrics = []

    dt = torch.tensor(config.learning.dt)
    for run_idx in range(runs):
        disturbance = float(np.random.uniform(-0.5, 0.5)) if random_disturbance else 0.0
        setpoint_val = random.randint(*setpoints_interval)
        setpoints = [torch.tensor(setpoint_val) for _ in range(steps)]

        Kp_, Ki_, Kd_ = system.tune_pid(dt=dt.item(), steps=steps, method=tuning_method)
        Kp, Ki, Kd = map(torch.tensor, (Kp_, Ki_, Kd_))
        pid.update_gains(Kp, Ki, Kd)

        pid.reset()
        system.reset()
        default_results = run_simulation_for_comparison(
            system=system,
            pid=pid,
            lstm_regulator=None,
            config=config,
            warm_up_steps=warm_up_steps,
            setpoints=setpoints,
        )

        pid.reset()
        system.reset()
        lstm_dynamic_results = run_simulation_for_comparison(
            system=system,
            pid=pid,
            lstm_regulator=lstm_regulator,
            config=config,
            warm_up_steps=warm_up_steps,
            setpoints=setpoints,
        )

        lstm_static_results = run_simulation_for_comparison(
            system=system,
            pid=pid,
            lstm_regulator=None,
            config=config,
            warm_up_steps=warm_up_steps,
            setpoints=setpoints,
        )

        default_m = calculate_metrics(
            default_results.positions, setpoints[-1], warm_up_steps, dt
        )
        lstm_dynamic_m = calculate_metrics(
            lstm_dynamic_results.positions, setpoints[-1], warm_up_steps, dt
        )
        lstm_static_m = calculate_metrics(
            lstm_static_results.positions, setpoints[-1], warm_up_steps, dt
        )

        default_metrics.append(default_m)
        lstm_dynamic_metrics.append(lstm_dynamic_m)
        lstm_static_metrics.append(lstm_static_m)

    print_aggregated_stats(default_metrics, "Default PID", runs)
    print_aggregated_stats(lstm_dynamic_metrics, "LSTM Dynamic PID", runs)
    print_aggregated_stats(lstm_static_metrics, "LSTM Static PID", runs)

    plot_aggregated_metrics_comparison(
        default_metrics=default_metrics,
        lstm_dynamic_metrics=lstm_dynamic_metrics,
        lstm_static_metrics=lstm_static_metrics,
        title_prefix="Porovnání PID regulátorů:",
        session_name=session_name,
    )


def print_aggregated_stats(
    metrics_list: List[Dict[str, float]], name: str, runs: int
) -> None:
    avg_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    std_metrics = {k: np.std([m[k] for m in metrics_list]) for k in metrics_list[0]}
    print(f"\n{name} Metrics over {runs} runs:")
    for k in avg_metrics:
        print(f"{k}: Mean={avg_metrics[k]:.4f}, Std={std_metrics[k]:.4f}")
