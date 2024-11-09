import torch
from torch import Tensor

from .base import BaseSystem


class SpringDamper(BaseSystem):
    def __init__(
        self, mass: Tensor, damping: Tensor, spring: Tensor, dt: Tensor
    ) -> None:
        """
        Mass-Spring-Damper System.

        Args:
            mass (float): Mass of the object (kg)
            damping (float): Damping coefficient (N·s/m)
            spring (float): Spring constant (N/m)
            dt (float): Time step (s)
        """
        self.mass = mass
        self.damping = damping
        self.spring = spring
        self.dt = dt

        self.position = torch.tensor(0.0)
        self.velocity = torch.tensor(0.0)
        self.acceleration = torch.tensor(0.0)

    def apply_control(
        self, control_output: Tensor, disturbance: Tensor = torch.tensor(0.0)
    ) -> Tensor:
        """
        Update the position and velocity based on the control output.

        Equation of motion:
            F_total = F_input - c*v - k*x - F_disturbance
            a = F_total / m
            v = v + a*dt
            x = x + v*dt

        Args:
            control_output (Tensor): External force applied to the mass (N)
            disturbance (Tensor): External disturbance force (N)

        Returns:
            Tensor: Updated position (m)
        """
        F_total = (
            control_output
            - self.damping * self.velocity
            - self.spring * self.position
            - disturbance
        )
        self.acceleration = F_total / self.mass
        self.velocity = self.velocity + self.acceleration * self.dt
        self.position = self.position + self.velocity * self.dt
        return self.position

    def reset(self) -> None:
        """Reset position, velocity, and acceleration to zero."""
        self.position = torch.tensor(0.0, dtype=torch.float32)
        self.velocity = torch.tensor(0.0, dtype=torch.float32)
        self.acceleration = torch.tensor(0.0, dtype=torch.float32)

    @property
    def X(self) -> Tensor:
        return self.position

    @property
    def dXdT(self) -> Tensor:
        return self.velocity

    @property
    def d2XdT2(self) -> Tensor:
        return self.acceleration

    @property
    def min_dt(self, oversampling_factor: float = 10.0) -> Tensor:
        """
        Calculate the minimum dt for good approximation based on the system's natural frequency
        and the Nyquist criterion.

        Args:
            oversampling_factor (float): Factor by which to oversample the Nyquist rate (default is 10)

        Returns:
            Tensor: Minimum dt value
        """
        # Calculate the natural frequency (omega_n)
        omega_n = torch.sqrt(self.spring / self.mass)

        # Apply the Nyquist criterion with oversampling
        min_dt = (torch.pi) / (oversampling_factor * omega_n)

        # Ensure stability for the Euler integration method
        max_stable_dt = 2.0 / omega_n

        # Choose the smaller dt to satisfy both criteria
        min_dt = torch.min(min_dt, max_stable_dt)

        return min_dt
