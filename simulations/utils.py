import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from config import cnfg


def simulate_system_response(system, num_steps: int, F: torch.Tensor):
    """
    Simulate the system response to a unit step input.

    Args:
        system: The system to simulate.
        num_steps (int): Number of simulation steps.
        F (torch.Tensor): Control input value.

    Returns:
        tuple: Arrays of time and response values.
    """
    responses = []
    time = np.linspace(0, num_steps * system.dt.item(), num_steps)
    for _ in range(num_steps):
        response = system.apply_control(F)
        responses.append(response.clone().detach())

    return time, responses


def plot_responses(time_responses, labels, title, ylabel, filename):
    """
    Plot the responses of different systems.

    Args:
        time_responses (list): List of tuples containing time and response arrays.
        labels (list): List of labels for the responses.
        title (str): Title of the plot.
        ylabel (str): Y-axis label.
        filename (str): Filename to save the plot.
    """
    plt.figure()
    for (time, response), label in zip(time_responses, labels):
        plt.plot(time, response, label=label)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(cnfg.SYSTEMS_PLOTS, filename)
    plt.savefig(plot_path)
    plt.show()


def plot_combined_phase_diagram(data, labels, title, filename):
    """
    Vykreslete kombinované fázové diagramy pro různé systémy.

    Argumenty:
        data (list): Seznam dvojic obsahujících časové a odezvové pole.
        labels (list): Seznam popisků pro odezvy.
        title (str): Název grafu.
        filename (str): Název souboru pro uložení grafu.
    """
    plt.figure()
    for (time, responses), label in zip(data, labels):
        time = torch.tensor(time)  # Převést na tensor
        responses = torch.tensor(responses)  # Převést na tensor
        changes = torch.diff(responses) / torch.diff(time)
        plt.plot(responses[:-1].numpy(), changes.numpy(), label=label)
    plt.xlabel("Pozice (m)" if "Pozice" in title else "Teplota (K)", fontsize=12)
    plt.ylabel(
        "Rychlost (m/s)" if "Pozice" in title else "Změna teploty (K/s)", fontsize=12
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(cnfg.SYSTEMS_PLOTS, filename)
    plt.savefig(plot_path, dpi=300)
    plt.show()
