import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from config import cnfg


def simulate_system_response(
        system, 
        num_steps: int, 
        F: torch.Tensor
        ):
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
        responses.append(response.item())

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
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(cnfg.SYSTEMS_PLOTS, filename)
    plt.savefig(plot_path)
    plt.show()
