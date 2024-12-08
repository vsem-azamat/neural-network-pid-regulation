import torch

from entities.systems import Trolley
from .utils import simulate_system_response, plot_responses, plot_combined_phase_diagram

if __name__ == "__main__":
    # Simulation parameters
    num_steps = 1000
    F_input = torch.tensor(3000.0)  # Unit step input (3000 N)

    # Parameters for stable (overdamped) response
    stable_trolley_system = Trolley(
        mass=torch.tensor(1000.0),  # Mass of the trolley (kg)
        spring=torch.tensor(100.0),  # Spring constant (N/m)
        friction=torch.tensor(800.0),  # High friction for overdamped response (N/(m/s))
        dt=torch.tensor(0.1),  # Time step (s)
    )
    stable_trolley_system.position = torch.tensor(0.0)  # Initial position (m)
    time_stable, positions_stable = simulate_system_response(
        stable_trolley_system, num_steps=num_steps, F=F_input
    )

    # Parameters for oscillatory (underdamped but stable) response
    oscillatory_trolley_system = Trolley(
        mass=torch.tensor(1000.0),  # Mass of the trolley (kg)
        spring=torch.tensor(100.0),  # Spring constant (N/m)
        friction=torch.tensor(
            100.0
        ),  # Moderate friction for underdamped response (N/(m/s))
        dt=torch.tensor(0.1),  # Time step (s)
    )
    oscillatory_trolley_system.position = torch.tensor(0.0)  # Initial position (m)
    time_oscillatory, positions_oscillatory = simulate_system_response(
        oscillatory_trolley_system, num_steps=num_steps, F=F_input
    )

    # Plot both responses on the same graph
    plot_responses(
        [(time_stable, positions_stable), (time_oscillatory, positions_oscillatory)],
        ["Stabilní odezva", "Oscilační odezva"],
        "Odezvy vozíku na jednotkový skok",
        "Pozice (m)",
        "response_trolley.pdf",
    )

    # Plot combined phase diagram
    plot_combined_phase_diagram(
        [(time_stable, positions_stable), (time_oscillatory, positions_oscillatory)],
        ["Stabilní odezva", "Oscilační odezva"],
        "Fázový diagram vozíku",
        "phase_trolley.pdf",
    )
