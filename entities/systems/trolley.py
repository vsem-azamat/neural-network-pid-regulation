"""Mass–spring–damper trolley: a second-order, oscillatory plant."""

import torch
from torch import Tensor

from .base import ZERO, BaseSystem


class Trolley(BaseSystem):
    """m·ẍ + c·ẋ + F_c·tanh(ẋ/ε) + k·x + k₃·x³ = F + d

    Integrated with semi-implicit (symplectic) Euler, which stays stable on an
    oscillatory plant for far larger time steps than explicit Euler.

    The two optional terms are what make gain scheduling a sensible thing to
    attempt on this plant at all:

    * ``spring_cubic`` (k₃) is a hardening spring. Local stiffness is
      k + 3·k₃·x², so with k=1 and k₃=0.02 the plant is 14 times stiffer at
      x=15 m than at the origin, and its natural frequency nearly quadruples
      across the operating range. One set of gains cannot suit both ends.
    * ``coulomb_friction`` (F_c) is dry friction, smoothed with tanh so the
      model stays differentiable. It is the classic reason a fixed controller
      leaves a steady-state offset: below the break-away force nothing moves,
      and the gain needed to overcome it is more than the gain that keeps the
      loop well damped once moving.

    Both default to zero, which recovers the linear plant.
    """

    def __init__(
        self,
        mass: Tensor,
        spring: Tensor,
        friction: Tensor,
        dt: Tensor,
        spring_cubic: Tensor | float = 0.0,
        coulomb_friction: Tensor | float = 0.0,
        stiction_velocity: float = 1e-2,
    ) -> None:
        """
        Args:
            mass: Trolley mass, kg.
            spring: Linear spring constant, N/m.
            friction: Viscous friction coefficient, N/(m/s).
            dt: Integration step, s.
            spring_cubic: Hardening coefficient k₃, N/m³.
            coulomb_friction: Dry friction force, N.
            stiction_velocity: Width of the tanh that smooths the dry-friction
                sign change, m/s. Small enough to behave like Coulomb friction,
                large enough to stay differentiable.
        """
        self.mass = torch.as_tensor(mass, dtype=torch.float32)
        self.spring = torch.as_tensor(spring, dtype=torch.float32)
        self.friction = torch.as_tensor(friction, dtype=torch.float32)
        self.dt = torch.as_tensor(dt, dtype=torch.float32)
        self.spring_cubic = torch.as_tensor(spring_cubic, dtype=torch.float32)
        self.coulomb_friction = torch.as_tensor(coulomb_friction, dtype=torch.float32)
        self.stiction_velocity = float(stiction_velocity)

        self.position = torch.tensor(0.0)
        self.velocity = torch.tensor(0.0)
        self.acceleration = torch.tensor(0.0)

    def apply_control(
        self, control_output: Tensor, disturbance: Tensor = ZERO
    ) -> Tensor:
        """Advance one step under force plus ``disturbance``, both in newtons."""
        force = control_output + disturbance
        restoring = self.spring * self.position + self.spring_cubic * self.position**3
        damping = self.friction * self.velocity
        if float(self.coulomb_friction) != 0.0:
            damping = damping + self.coulomb_friction * torch.tanh(
                self.velocity / self.stiction_velocity
            )
        self.acceleration = (force - damping - restoring) / self.mass
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

    def local_stiffness(self, amplitude: float = 0.0) -> Tensor:
        """Tangent stiffness k + 3·k₃·x² at a displacement of ``amplitude``."""
        return self.spring + 3.0 * self.spring_cubic * float(amplitude) ** 2

    def min_dt(self, oversampling_factor: float = 10.0, amplitude: float = 0.0) -> Tensor:
        """Sampling step that resolves the damped natural frequency.

        Takes the smaller of the oversampled Nyquist step and the explicit-Euler
        stability bound 2/ω_n, so the number is safe for either integrator.

        ``amplitude`` matters once the spring hardens: the plant gets faster the
        further it travels, so a step size checked only at the origin can be far
        too large where the controller actually operates.
        """
        omega_n = torch.sqrt(self.local_stiffness(amplitude) / self.mass)
        nyquist_dt = torch.pi / (oversampling_factor * omega_n)
        max_stable_dt = 2.0 / omega_n
        return torch.min(nyquist_dt, max_stable_dt)

    @property
    def damping_ratio(self) -> Tensor:
        """ζ < 1 underdamped, ζ = 1 critical, ζ > 1 overdamped.

        Reported for the linear part of the spring, at the origin.
        """
        return self.friction / (2.0 * torch.sqrt(self.spring * self.mass))

    @property
    def is_nonlinear(self) -> bool:
        return float(self.spring_cubic) != 0.0 or float(self.coulomb_friction) != 0.0
