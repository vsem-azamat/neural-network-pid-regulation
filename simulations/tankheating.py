import torch

from entities.systems import TankHeating
from .utils import simulate_system_response, plot_responses, plot_combined_phase_diagram

if __name__ == "__main__":
    # Simulation parameters
    num_steps = 1000
    Tq_input = torch.tensor(400.0)  # Target temperature (400 K)

    # Parameters for the tank heating system
    tank_heating_system = TankHeating(dt=torch.tensor(0.05))  # Time step (s)
    tank_heating_system.T = torch.tensor(300.0)  # Initial temperature (K)
    tank_heating_system.Tf = torch.tensor(350.0)  # Temperature of the incoming fluid (K)
    tank_heating_system.epsilon = torch.tensor(0.5)  # Heat capacity ratio
    tank_heating_system.tau = torch.tensor(2.0)  # Residence time (s)
    tank_heating_system.Q = torch.tensor(5.0)  # Heat transfer coefficient

    time, temperatures = simulate_system_response(
        tank_heating_system, num_steps=num_steps, F=Tq_input
    )

    # Plot the temperature response
    plot_responses(
        [(time, temperatures)],
        ["Teplotní odezva"],
        "Odezva ohřevu nádrže na cílovou teplotu",
        "Teplota (K)",
        "response_tankheating.pdf",
    )

    # Plot the phase diagram
    plot_combined_phase_diagram(
        [(time, temperatures)],
        ["Teplotní odezva"],
        "Fázový diagram ohřevu nádrže",
        "phase_tankheating.pdf",
    )
