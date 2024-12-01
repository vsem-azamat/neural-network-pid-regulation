import torch

from entities.systems import Thermal
from .utils import simulate_system_response, plot_responses


if __name__ == "__main__":
    # Initialize the Thermal system with specific parameters
    thermal_system = Thermal(
        thermal_capacity=torch.tensor(500.0),  # Thermal capacity (J/K)
        heat_transfer_coefficient=torch.tensor(10.0),  # Heat transfer coefficient (W/K)
        dt=torch.tensor(0.1),  # Time step (s)
    )
    thermal_system.temperature = torch.tensor(293.15)  # Initial temperature (20°C in Kelvin)
    F = torch.tensor(3000.0)  # Unit step input (3000 W)
    # Simulate the response of the thermal system to a unit step input
    time, temperatures = simulate_system_response(thermal_system, num_steps=5000, F=F)

    # Plot the response
    plot_responses(
        [(time, temperatures)],
        [f'Vstupní výkon: {F.item()} W'],
        f'Odezva tepelného systému na jednotkový skok: {F.item()} W',
        'Teplota (K)',
        'thermal_response.pdf'
    )
