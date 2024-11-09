import torch
from torch import Tensor

from .base import BaseSystem


class Thermal(BaseSystem):
    def __init__(
        self, thermal_capacity: Tensor, heat_transfer_coefficient: Tensor, dt: Tensor
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
        self.temperature = torch.tensor(293.15)  # Starting at 20°C in Kelvin
        self.temp_derivative = torch.tensor(0.0)  # dT/dt

    def apply_control(
        self, control_output: Tensor, disturbance: Tensor = torch.tensor(0.0)
    ) -> Tensor:
        """
        Update the temperature of the system based on the control output (heat input)

        Equation of model:
            dT/dt = (Q - h*T + disturbance) / C
            T(t+dt) = T(t) + dT/dt * dt
        where T is the temperature, Q is the heat input, h is the heat transfer coefficient,
        and C is the thermal capacity
        """
        temp_derivative = (
            control_output
            - self.heat_transfer_coefficient * self.temperature
            + disturbance
        ) / self.thermal_capacity
        self.temp_derivative = temp_derivative
        self.temperature = self.temperature + temp_derivative * self.dt
        return self.temperature

    def reset(self) -> None:
        """Reset the system temperature to initial conditions"""
        self.temperature = torch.tensor(293.15, dtype=torch.float32)
        self.temp_derivative = torch.tensor(0.0, dtype=torch.float32)

    @property
    def X(self) -> Tensor:
        return self.temperature

    @property
    def dXdT(self) -> Tensor:
        return self.temp_derivative

    @property
    def d2XdT2(self) -> Tensor:
        return torch.tensor(0.0)

    @property
    def min_dt(self, oversampling_factor: float = 10.0) -> Tensor:
        """
        Calculate the minimum dt for good approximation based on the system's time constant
        and the Nyquist criterion.

        Args:
            oversampling_factor (float): Factor by which to oversample the Nyquist rate (default is 10)

        Returns:
            Tensor: Minimum dt value
        """
        # Calculate the system's time constant (tau)
        tau = self.thermal_capacity / self.heat_transfer_coefficient

        # The system's cutoff frequency (omega_c)
        omega_c = 1.0 / tau

        # Apply the Nyquist criterion with oversampling
        min_dt = (torch.pi) / (oversampling_factor * omega_c)

        # Ensure stability for the Euler integration method
        max_stable_dt = 2.0 * tau  # Stability condition for first-order system

        # Choose the smaller dt to satisfy both criteria
        min_dt = torch.min(min_dt, max_stable_dt)

        return min_dt
