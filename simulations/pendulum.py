import torch

from entities.systems import NonLinearPendulumCart
from .utils import simulate_system_response, plot_responses, plot_combined_phase_diagram

if __name__ == "__main__":
    # Simulation parameters
    num_steps = 500
    control_input = torch.tensor(10.0)  # Control force (10 N)

    # Parameters for stable (overdamped) response
    stable_pendulum_cart_system = NonLinearPendulumCart(
        cart_mass=torch.tensor(5.0),  # Mass of the cart (kg)
        pendulum_mass=torch.tensor(0.5),  # Mass of the pendulum bob (kg)
        pendulum_length=torch.tensor(1.0),  # Length of the pendulum (m)
        friction=torch.tensor(10.0),  # High friction for overdamped response (N·s/m)
        gravity=torch.tensor(9.81),  # Gravitational acceleration (m/s^2)
        dt=torch.tensor(0.01),  # Time step (s)
    )
    stable_pendulum_cart_system.reset()  # Reset to initial conditions
    time_stable, states_stable = simulate_system_response(
        stable_pendulum_cart_system, num_steps=num_steps, F=control_input
    )

    # Parameters for oscillatory (underdamped but stable) response
    oscillatory_pendulum_cart_system = NonLinearPendulumCart(
        cart_mass=torch.tensor(1.0),  # Mass of the cart (kg)
        pendulum_mass=torch.tensor(0.1),  # Mass of the pendulum bob (kg)
        pendulum_length=torch.tensor(1.0),  # Length of the pendulum (m)
        friction=torch.tensor(0.1),  # Low friction for underdamped response (N·s/m)
        gravity=torch.tensor(9.81),  # Gravitational acceleration (m/s^2)
        dt=torch.tensor(0.01),  # Time step (s)
    )
    oscillatory_pendulum_cart_system.reset()  # Reset to initial conditions
    time_oscillatory, states_oscillatory = simulate_system_response(
        oscillatory_pendulum_cart_system, num_steps=num_steps, F=control_input
    )

    # Extract positions and angles from the state vector
    positions_stable = [state[0].item() for state in states_stable]
    angles_stable = [state[2].item() for state in states_stable]
    positions_oscillatory = [state[0].item() for state in states_oscillatory]
    angles_oscillatory = [state[2].item() for state in states_oscillatory]

    # Plot both position responses on the same graph
    plot_responses(
        [(time_stable, positions_stable), (time_oscillatory, positions_oscillatory)],
        ["Stabilní odezva", "Oscilační odezva"],
        "Odezva vozíku na řídicí sílu",
        "Pozice (m)",
        "response_pendulum_cart_position.pdf",
    )

    # Plot both angle responses on the same graph
    plot_responses(
        [(time_stable, angles_stable), (time_oscillatory, angles_oscillatory)],
        ["Stabilní odezva", "Oscilační odezva"],
        "Odezva kyvadla na řídicí sílu",
        "Úhel (rad)",
        "response_pendulum_cart_angle.pdf",
    )

    # Plot combined phase diagram for positions
    plot_combined_phase_diagram(
        [(time_stable, positions_stable), (time_oscillatory, positions_oscillatory)],
        ["Stabilní odezva", "Oscilační odezva"],
        "Fázový diagram vozíku",
        "phase_pendulum_cart_position.pdf",
    )

    # Plot combined phase diagram for angles
    plot_combined_phase_diagram(
        [(time_stable, angles_stable), (time_oscillatory, angles_oscillatory)],
        ["Stabilní odezva", "Oscilační odezva"],
        "Fázový diagram kyvadla",
        "phase_pendulum_cart_angle.pdf",
    )
