import torch

from entities.systems import SpringDamper
from .utils import simulate_system_response, plot_responses


if __name__ == "__main__":
    # Simulation parameters
    num_steps = 300
    F_input = torch.tensor(100.0)  # Unit step input (100 N)

    # Parameters for overdamped response
    overdamped_springdamper_system = SpringDamper(
        mass=torch.tensor(10.0),      # Mass (kg)
        damping=torch.tensor(50.0),   # High damping coefficient (N·s/m)
        spring=torch.tensor(20.0),    # Spring constant (N/m)
        dt=torch.tensor(0.1),         # Time step (s)
    )
    overdamped_springdamper_system.position = torch.tensor(0.0)  # Initial position (m)
    time_overdamped, positions_overdamped = simulate_system_response(
        overdamped_springdamper_system, num_steps=num_steps, F=F_input)

    # Parameters for underdamped response
    underdamped_springdamper_system = SpringDamper(
        mass=torch.tensor(10.0),      # Mass (kg)
        damping=torch.tensor(5.0),    # Low damping coefficient (N·s/m)
        spring=torch.tensor(20.0),    # Spring constant (N/m)
        dt=torch.tensor(0.1),         # Time step (s)
    )
    underdamped_springdamper_system.position = torch.tensor(0.0)  # Initial position (m)
    time_underdamped, positions_underdamped = simulate_system_response(
        underdamped_springdamper_system, num_steps=num_steps, F=F_input)

    # Plot both responses on the same graph
    plot_responses(
        [(time_overdamped, positions_overdamped), (time_underdamped, positions_underdamped)],
        ['Přetlumená odezva', 'Podtlumená odezva'],
        'Odezvy pružinového tlumiče na jednotkový skok',
        'Pozice (m)',
        'springdamper_response.pdf'
    )
