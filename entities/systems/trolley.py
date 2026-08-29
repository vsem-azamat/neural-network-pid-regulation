"""Mass–spring–damper trolley: a second-order, oscillatory plant."""

import torch
from torch import Tensor

from .base import ZERO, BaseSystem


class Trolley(BaseSystem):
    """m·ẍ + c·ẋ + k·x = F + d

    Integrated with semi-implicit (symplectic) Euler, which stays stable on an
    oscillatory plant for far larger time steps than explicit Euler.
    """

    def __init__(
        self, mass: Tensor, spring: Tensor, friction: Tensor, dt: Tensor
    ) -> None:
        """
        Args:
            mass: Trolley mass, kg.
            spring: Spring constant, N/m.
            friction: Viscous friction coefficient, N/(m/s).
            dt: Integration step, s.
        """
        self.mass = torch.as_tensor(mass, dtype=torch.float32)
        self.spring = torch.as_tensor(spring, dtype=torch.float32)
        self.friction = torch.as_tensor(friction, dtype=torch.float32)
        self.dt = torch.as_tensor(dt, dtype=torch.float32)

        self.position = torch.tensor(0.0)
        self.velocity = torch.tensor(0.0)
        self.acceleration = torch.tensor(0.0)

    def apply_control(
        self, control_output: Tensor, disturbance: Tensor = ZERO
    ) -> Tensor:
        """Advance one step under force plus ``disturbance``, both in newtons."""
        force = control_output + disturbance
        self.acceleration = (
            force - self.friction * self.velocity - self.spring * self.position
        ) / self.mass
        # Semi-implicit Euler: velocity first, then position from the *new* velocity.
        self.velocity = self.velocity + self.acceleration * self.dt
        self.position = self.position + self.velocity * self.dt
        return self.position

    def reset(self) -> None:
        self.position = torch.tensor(0.0)
        self.velocity = torch.tensor(0.0)
        self.acceleration = torch.tensor(0.0)

    def detach_state(self) -> None:
        self.position = self.position.detach()
        self.velocity = self.velocity.detach()
        self.acceleration = self.acceleration.detach()

    @property
    def X(self) -> Tensor:
        return self.position

    @property
    def dXdT(self) -> Tensor:
        return self.velocity

    @property
    def d2XdT2(self) -> Tensor:
        return self.acceleration

    def min_dt(self, oversampling_factor: float = 10.0) -> Tensor:
        """Sampling step that resolves the damped natural frequency.

        Takes the smaller of the oversampled Nyquist step and the explicit-Euler
        stability bound 2/ω_n, so the number is safe for either integrator.
        """
        omega_n = torch.sqrt(self.spring / self.mass)
        nyquist_dt = torch.pi / (oversampling_factor * omega_n)
        max_stable_dt = 2.0 / omega_n
        return torch.min(nyquist_dt, max_stable_dt)

    @property
    def damping_ratio(self) -> Tensor:
        """ζ < 1 underdamped, ζ = 1 critical, ζ > 1 overdamped."""
        return self.friction / (2.0 * torch.sqrt(self.spring * self.mass))
