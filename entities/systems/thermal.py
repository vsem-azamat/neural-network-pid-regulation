import torch
from torch import Tensor

from .base import BaseSystem

class Thermal(BaseSystem):
    def __init__(self, 
                 thermal_capacity: Tensor,
                 heat_transfer_coefficient: Tensor,
                 dt: Tensor
                ) -> None:
        """
        A simple first-order thermal system.

        Args:
            thermal_capacity (float): thermal capacity of the system (J/K)
            heat_transfer_coefficient (float): heat transfer coefficient (W/K)
            dt (float): time step between the current and previous state (s)
        """
        self.thermal_capacity = thermal_capacity
        self.heat_transfer_coefficient = heat_transfer_coefficient
        self.dt = dt

        self.temperature = torch.tensor(0.)  # Initial temperature in Kelvin above ambient

    def apply_control(self, control_output: Tensor, disturbance: Tensor = torch.tensor(0.)) -> Tensor:
        """
        Update the temperature of the system based on the control output (heat input)
        
        Equation of model:
            dT/dt = (Q - h*T) / C
            T(t+dt) = T(t) + dT/dt * dt
        where T is the temperature, Q is the heat input, h is the heat transfer coefficient,
        and C is the thermal capacity
        """
        assert control_output is not None, "Control output is None"
        
        # Calculate the temperature derivative
        temp_derivative = (control_output - self.heat_transfer_coefficient * self.temperature) / self.thermal_capacity + disturbance
        
        # Update the temperature using Euler integration
        self.temperature = self.temperature + temp_derivative * self.dt
        
        return self.temperature

    def get_state(self) -> Tensor:
        return self.temperature

    def reset(self) -> None:
        """Reset the system temperature to initial conditions"""
        self.temperature = torch.tensor(0., dtype=torch.float32)
