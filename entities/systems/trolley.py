import torch
from torch import Tensor

from .base import BaseSystem


class Trolley(BaseSystem):
    def __init__(
        self, mass: Tensor, spring: Tensor, friction: Tensor, dt: Tensor
    ) -> None:
        """
        Args:
                mass (float): mass of the trolley
                spring (float): spring constant of the trolley
                friction (float): friction coefficient of the trolley
                dt (float): time step between the current and previous position
        """
        self.mass = mass
        self.friction = friction
        self.spring = spring
        self.dt = dt

        self.position = torch.tensor(0.0)
        self.delta_position = torch.tensor(0.0)
        self.velocity = torch.tensor(0.0)
        self.acceleration = torch.tensor(0.0)

    def apply_control(
        self, control_output: Tensor, disturbance: Tensor = torch.tensor(0.0)
    ) -> Tensor:
        """
            Update the position and velocity of the trolley based on the control output

            Equation of model:
        F = ma
        a = F/m
        a = F/m - friction*v/m - spring_constant*x/m - disturbance/m
        v = v + a*dt
        x = x + v*dt
        """
        assert control_output is not None, "Control output is None"
        F = control_output
        self.acceleration = (
            F / self.mass
            - self.friction * self.velocity.clone().detach() / self.mass
            - self.spring * self.position.clone().detach() / self.mass
            - disturbance / self.mass
        )
        self.velocity = self.velocity.clone().detach() + self.acceleration * self.dt
        self.position = self.position.clone().detach() + self.velocity * self.dt
        return self.position

    def reset(self) -> None:
        """Reset: position, velocity, and delta_position to zero"""
        self.position = torch.tensor(0)
        self.velocity = torch.tensor(0)
        self.delta_position = torch.tensor(0)
        self.acceleration = torch.tensor(0)

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
