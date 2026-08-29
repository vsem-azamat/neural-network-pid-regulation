"""Discrete PID controller in several numerical formulations.

Two properties matter for the rest of the project and are easy to lose:

1. **The controller is differentiable.**  Gains arrive from the LSTM as tensors
   carrying an autograd graph; the control signal must carry that graph through
   to the plant so the tracking loss can be back-propagated into the network.
   Internal state is therefore *not* detached here — the training loop truncates
   the graph explicitly at window boundaries (see ``PID.detach_state``).

2. **Saturation feeds back into the state.**  Clamping only the returned value
   while the integrator (or, in the incremental form, ``u_k_1``) keeps
   accumulating is textbook integral windup: the plant sits against its limit
   while the controller's internal state runs away, and the output stays stuck
   long after the error has changed sign.
"""

from typing import Literal, Optional

import torch
from torch import Tensor

Method = Literal[
    "standard",
    "backward_euler",
    "trapezoidal",
    "forward_euler",
    "bilinear_transform",
]


class PID:
    def __init__(
        self, initial_KP: Tensor, initial_KI: Tensor, initial_KD: Tensor
    ) -> None:
        self.Kp = initial_KP
        self.Ki = initial_KI
        self.Kd = initial_KD

        # Error history for the incremental (velocity) form.
        self.e_k = torch.tensor(0.0)
        self.e_k_1 = torch.tensor(0.0)
        self.e_k_2 = torch.tensor(0.0)
        self.u_k_1 = torch.tensor(0.0)

        # State for the positional forms.
        self.integral = torch.tensor(0.0)
        self.prev_error = torch.tensor(0.0)

        self.saturation_max: Optional[Tensor] = None
        self.saturation_min: Optional[Tensor] = None

    # ── gain access ──────────────────────────────────────────────────────
    @property
    def E(self) -> Tensor:
        return self.e_k

    @property
    def dE(self) -> Tensor:
        return self.e_k - self.e_k_1

    @property
    def gains(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.Kp, self.Ki, self.Kd

    def update_gains(self, new_Kp, new_Ki, new_Kd) -> None:
        self.Kp = torch.as_tensor(new_Kp)
        self.Ki = torch.as_tensor(new_Ki)
        self.Kd = torch.as_tensor(new_Kd)

    def set_limits(self, max_limit: Tensor, min_limit: Tensor) -> None:
        assert max_limit > min_limit, "Max limit must be greater than min limit"
        self.saturation_max = torch.as_tensor(max_limit)
        self.saturation_min = torch.as_tensor(min_limit)

    # ── computation ──────────────────────────────────────────────────────
    def compute(self, error: Tensor, dt: Tensor, method: Method = "standard") -> Tensor:
        match method:
            case "standard":
                return self.compute_standard(error, dt)
            case "backward_euler":
                return self.compute_backward_euler(error, dt)
            case "trapezoidal":
                return self.compute_trapezoidal(error, dt)
            case "forward_euler":
                return self.compute_forward_euler(error, dt)
            case "bilinear_transform":
                return self.compute_bilinear_transform(error, dt)
            case _:
                raise ValueError(
                    f"Unknown method {method!r}. Choose one of: standard, "
                    "backward_euler, trapezoidal, forward_euler, bilinear_transform"
                )

    def _saturate(self, u_k: Tensor) -> Tensor:
        if self.saturation_max is None or self.saturation_min is None:
            return u_k
        return torch.clamp(u_k, self.saturation_min, self.saturation_max)

    def compute_standard(self, error: Tensor, dt: Tensor) -> Tensor:
        """Incremental (velocity) form: u_k = u_{k-1} + Δu_k."""
        self.e_k_2 = self.e_k_1
        self.e_k_1 = self.e_k
        self.e_k = error

        u_k = (
            self.u_k_1
            + self.Kp * (self.e_k - self.e_k_1)
            + self.Ki * self.e_k * dt
            + self.Kd * ((self.e_k - self.e_k_1) - (self.e_k_1 - self.e_k_2)) / dt
        )

        # Store the *saturated* output: in the incremental form u_{k-1} is the
        # integrator, so an unclamped value here is exactly integral windup.
        u_k = self._saturate(u_k)
        self.u_k_1 = u_k
        return u_k

    def _positional(self, error: Tensor, integral: Tensor, dt: Tensor) -> Tensor:
        """Shared tail of the positional forms, with conditional integration."""
        derivative = (error - self.prev_error) / dt
        u_unclamped = self.Kp * error + self.Ki * integral + self.Kd * derivative
        u_k = self._saturate(u_unclamped)

        # Conditional integration (clamping anti-windup): only commit the new
        # integral if the controller is not saturated, or if the error is
        # driving the output back inside the limits.
        saturated = not torch.equal(u_k.detach(), u_unclamped.detach())
        unwinding = bool((u_unclamped.detach() * error.detach() < 0).all())
        if not saturated or unwinding:
            self.integral = integral

        self.prev_error = error
        self.e_k = error
        return u_k

    def compute_backward_euler(self, error: Tensor, dt: Tensor) -> Tensor:
        return self._positional(error, self.integral + error * dt, dt)

    def compute_forward_euler(self, error: Tensor, dt: Tensor) -> Tensor:
        return self._positional(error, self.integral + self.prev_error * dt, dt)

    def compute_trapezoidal(self, error: Tensor, dt: Tensor) -> Tensor:
        integral = self.integral + (error + self.prev_error) * dt / 2
        return self._positional(error, integral, dt)

    def compute_bilinear_transform(self, error: Tensor, dt: Tensor) -> Tensor:
        # Tustin discretisation of the integral term; identical to trapezoidal
        # integration, kept separate because the thesis discusses them apart.
        integral = self.integral + dt / 2 * (error + self.prev_error)
        return self._positional(error, integral, dt)

    # ── lifecycle ────────────────────────────────────────────────────────
    def detach_state(self) -> None:
        """Truncate the autograd graph at a TBPTT window boundary.

        Keeps the numeric state (so the simulation is continuous) but drops the
        history that backward() would otherwise have to walk through.
        """
        self.e_k = self.e_k.detach()
        self.e_k_1 = self.e_k_1.detach()
        self.e_k_2 = self.e_k_2.detach()
        self.u_k_1 = self.u_k_1.detach()
        self.integral = self.integral.detach()
        self.prev_error = self.prev_error.detach()

    def reset(self) -> None:
        self.integral = torch.tensor(0.0)
        self.prev_error = torch.tensor(0.0)
        self.e_k = torch.tensor(0.0)
        self.e_k_1 = torch.tensor(0.0)
        self.e_k_2 = torch.tensor(0.0)
        self.u_k_1 = torch.tensor(0.0)
