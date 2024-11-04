import torch
from torch import Tensor

from .base import BaseSystem


class NonLinearPendulumCart(BaseSystem):
    def __init__(
        self,
        cart_mass: Tensor,
        pendulum_mass: Tensor,
        pendulum_length: Tensor,
        friction: Tensor,
        gravity: Tensor,
        dt: Tensor,
    ) -> None:
        """
        Args:
            cart_mass (float): mass of the cart
            pendulum_mass (float): mass of the pendulum bob
            pendulum_length (float): length of the pendulum
            friction (float): friction coefficient of the cart
            gravity (float): gravitational acceleration
            dt (float): time step between the current and previous state
        """
        self.M = cart_mass
        self.m = pendulum_mass
        self.L = pendulum_length
        self.b = friction
        self.g = gravity
        self.dt = dt

        self.x = torch.tensor(0.0)  # cart position
        self.x_dot = torch.tensor(0.0)  # cart velocity
        self.theta = torch.tensor(0.0)  # pendulum angle
        self.theta_dot = torch.tensor(0.0)  # pendulum angular velocity

    def apply_control(
        self, control_output: Tensor, disturbance: Tensor = torch.tensor(0.0)
    ) -> Tensor:
        """
        Update the state of the system based on the control output

        Equations of motion:
        (M+m)x'' + b*x' + m*L*θ''*cos(θ) - m*L*θ'²*sin(θ) = F + d
        (I+m*L²)θ'' + m*g*L*sin(θ) + m*L*x''*cos(θ) = 0

        where x is cart position, θ is pendulum angle, F is control force, and d is disturbance
        """
        assert control_output is not None, "Control output is None"
        F = control_output + disturbance

        # Simplify notation
        x, x_dot = self.x, self.x_dot
        theta, theta_dot = self.theta, self.theta_dot
        M, m, L, b, g = self.M, self.m, self.L, self.b, self.g

        # Compute accelerations
        denominator = M + m * torch.sin(theta) ** 2
        x_ddot = (
            F
            + m * L * theta_dot**2 * torch.sin(theta)
            - b * x_dot
            - m * g * torch.sin(theta) * torch.cos(theta)
        ) / denominator
        theta_ddot = (-g * torch.sin(theta) - x_ddot * torch.cos(theta)) / L

        # Update state using Euler integration
        self.x = x + x_dot * self.dt
        self.x_dot = x_dot + x_ddot * self.dt
        self.theta = theta + theta_dot * self.dt
        self.theta_dot = theta_dot + theta_ddot * self.dt

        return self.get_X()

    def get_X(self) -> Tensor:
        return torch.tensor([self.x, self.x_dot, self.theta, self.theta_dot])

    def get_position(self) -> Tensor:
        return self.x

    def get_pendulum_tip_position(self) -> Tensor:
        x_tip = self.x + self.L * torch.sin(self.theta)
        y_tip = self.L * torch.cos(self.theta)
        return torch.tensor([x_tip, y_tip])

    def reset(self) -> None:
        """Reset the system state to initial conditions"""
        self.x = torch.tensor(0.0, dtype=torch.float32)
        self.x_dot = torch.tensor(0.0, dtype=torch.float32)
        self.theta = torch.tensor(0.1, dtype=torch.float32)  # Small initial angle
        self.theta_dot = torch.tensor(0.0, dtype=torch.float32)
