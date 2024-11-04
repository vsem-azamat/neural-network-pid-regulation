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
        self, control_output: Tensor, distrubance: Tensor = torch.tensor(0.0)
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
            - distrubance / self.mass
        )
        self.velocity = self.velocity.clone().detach() + self.acceleration * self.dt
        self.position = self.position.clone().detach() + self.velocity * self.dt
        return self.position

    @property
    def X(self) -> Tensor:
        return self.position

    @property
    def dXdT(self) -> Tensor:
        return self.velocity

    @property
    def d2XdT2(self) -> Tensor:
        return self.acceleration

    def get_U(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.position, self.position / self.dt, self.position / (self.dt**2)

    def reset(self) -> None:
        """Reset: position, velocity, and delta_position to zero"""
        self.position = torch.tensor(0, dtype=torch.float32)
        self.velocity = torch.tensor(0, dtype=torch.float32)
        self.delta_position = torch.tensor(0, dtype=torch.float32)
        self.acceleration = torch.tensor(0, dtype=torch.float32)
