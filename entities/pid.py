import torch
from torch import Tensor
from typing import Literal


class PID:
    def __init__(self, initial_KP: Tensor, initial_KI: Tensor, initial_KD: Tensor):
        self.Kp = initial_KP
        self.Ki = initial_KI
        self.Kd = initial_KD

        # PID states
        self.e_k = torch.tensor(0.0)
        self.e_k_1 = torch.tensor(0.0)
        self.e_k_2 = torch.tensor(0.0)
        self.u_k_1 = torch.tensor(0.0)

        # Integral and previous error states
        self.integral = torch.tensor(0.0)
        self.prev_error = torch.tensor(0.0)

        # PID saturation limits
        self.saturation_max = None
        self.saturation_min = None

    @property
    def E(self) -> Tensor:
        return self.e_k

    @property
    def dE(self) -> Tensor:
        return self.e_k - self.e_k_1

    def update_gains(self, new_Kp: Tensor, new_Ki: Tensor, new_Kd: Tensor) -> None:
        self.Kp = new_Kp
        self.Ki = new_Ki
        self.Kd = new_Kd

    def compute(
        self,
        error: Tensor,
        dt: Tensor,
        method: Literal[
            "standard",
            "backward_euler",
            "trapezoidal",
            "forward_euler",
            "bilinear_transform",
        ] = "standard",
    ) -> Tensor:
        # return self.compute_standard(error, dt)
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
                    "Invalid method. Choose between 'standard', 'backward_euler', 'trapezoidal', 'forward_euler', 'bilinear_transform'"
                )

    def compute_standard(self, error: Tensor, dt: Tensor) -> Tensor:
        self.e_k_2 = self.e_k_1
        self.e_k_1 = self.e_k
        self.e_k = error

        u_k = (
            self.u_k_1
            + self.Kp * (self.e_k - self.e_k_1)
            + self.Ki * self.e_k * dt
            + self.Kd * ((self.e_k - self.e_k_1) - (self.e_k_1 - self.e_k_2)) / dt
        )

        self.u_k_1 = u_k.clone().detach()

        if isinstance(self.saturation_max, Tensor) and isinstance(
            self.saturation_min, Tensor
        ):
            return torch.clamp(u_k, self.saturation_min, self.saturation_max)
        return u_k

    def compute_backward_euler(self, error: Tensor, dt: Tensor) -> Tensor:
        self.e_k = error
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        u_k = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        if isinstance(self.saturation_max, Tensor) and isinstance(
            self.saturation_min, Tensor
        ):
            return torch.clamp(u_k, self.saturation_min, self.saturation_max)
        return u_k

    def compute_trapezoidal(self, error: Tensor, dt: Tensor) -> Tensor:
        self.e_k = error
        self.integral += (error + self.prev_error) * dt / 2
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        u_k = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        if isinstance(self.saturation_max, Tensor) and isinstance(
            self.saturation_min, Tensor
        ):
            return torch.clamp(u_k, self.saturation_min, self.saturation_max)
        return u_k

    def compute_forward_euler(self, error: Tensor, dt: Tensor) -> Tensor:
        self.e_k = error
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        u_k = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        if isinstance(self.saturation_max, Tensor) and isinstance(
            self.saturation_min, Tensor
        ):
            return torch.clamp(u_k, self.saturation_min, self.saturation_max)
        return u_k

    def compute_bilinear_transform(self, error: Tensor, dt: Tensor) -> Tensor:
        self.e_k = error
        self.integral = self.integral + dt / 2 * (error + self.prev_error)
        derivative = 2 * (error - self.prev_error) / dt - self.u_k_1
        self.prev_error = error

        u_k = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        if isinstance(self.saturation_max, Tensor) and isinstance(
            self.saturation_min, Tensor
        ):
            return torch.clamp(u_k, self.saturation_min, self.saturation_max)
        return u_k

    def set_limits(self, max_limit: Tensor, min_limit: Tensor) -> None:
        assert max_limit > min_limit, "Max limit must be greater than min limit"
        self.saturation_max = max_limit
        self.saturation_min = min_limit
