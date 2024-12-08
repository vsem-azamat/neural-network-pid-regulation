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
        Nonlinear Pendulum-Cart System.

        Args:
            cart_mass (float): mass of the cart (kg)
            pendulum_mass (float): mass of the pendulum bob (kg)
            pendulum_length (float): length of the pendulum (m)
            friction (float): friction coefficient of the cart (N·s/m)
            gravity (float): gravitational acceleration (m/s^2)
            dt (float): time step between the current and previous state (s)
        """
        self.M = cart_mass
        self.m = pendulum_mass
        self.L = pendulum_length
        self.b = friction
        self.g = gravity
        self.dt = dt

        # Initialize state variables
        self.x = torch.tensor(0.0)  # Cart position (m)
        self.x_dot = torch.tensor(0.0)  # Cart velocity (m/s)
        self.theta = torch.tensor(0.1)  # Pendulum angle (rad), small initial angle
        self.theta_dot = torch.tensor(0.0)  # Pendulum angular velocity (rad/s)

    def apply_control(
        self, control_output: Tensor, distrubance: Tensor = torch.tensor(0.0)
    ) -> Tensor:
        """
        Update the state of the system based on the control output.

        Args:
            control_output (Tensor): Control force applied to the cart (N)
            disturbance (Tensor): External disturbance force (N)

        Returns:
            Tensor: Current state vector [x, x_dot, theta, theta_dot]
        """
        assert control_output is not None, "Control output is None"
        F = control_output + distrubance

        # Simplify notation
        x, x_dot = self.x, self.x_dot
        theta, theta_dot = self.theta, self.theta_dot
        M, m, L, b, g = self.M, self.m, self.L, self.b, self.g

        # Compute accelerations using the equations of motion
        denom = M + m * torch.sin(theta) ** 2

        x_ddot = (
            F
            + m * L * theta_dot**2 * torch.sin(theta)
            - b * x_dot
            - m * g * torch.sin(theta) * torch.cos(theta)
        ) / denom

        theta_ddot = (-g * torch.sin(theta) - x_ddot * torch.cos(theta)) / L

        # Update state using Euler integration
        self.x = x + x_dot * self.dt
        self.x_dot = x_dot + x_ddot * self.dt
        self.theta = theta + theta_dot * self.dt
        self.theta_dot = theta_dot + theta_ddot * self.dt

        return self.X

    def reset(self) -> None:
        """Reset the system state to initial conditions."""
        self.x = torch.tensor(0.0)
        self.x_dot = torch.tensor(0.0)
        self.theta = torch.tensor(0.1)  # Small initial angle
        self.theta_dot = torch.tensor(0.0)

    @property
    def X(self) -> Tensor:
        """State vector: [x, x_dot, theta, theta_dot]."""
        return torch.tensor([self.x, self.x_dot, self.theta, self.theta_dot])

    @property
    def dXdT(self) -> Tensor:
        """First derivative of the state vector."""
        # x_dot, x_ddot, theta_dot, theta_ddot
        x, x_dot, theta, theta_dot = self.x, self.x_dot, self.theta, self.theta_dot
        M, m, L, b, g = self.M, self.m, self.L, self.b, self.g

        denom = M + m * torch.sin(theta) ** 2

        x_ddot = (
            -b * x_dot
            - m * g * torch.sin(theta) * torch.cos(theta)
            + m * L * theta_dot**2 * torch.sin(theta)
        ) / denom

        theta_ddot = (-g * torch.sin(theta) - x_ddot * torch.cos(theta)) / L

        return torch.tensor([x_dot, x_ddot, theta_dot, theta_ddot])

    @property
    def d2XdT2(self) -> Tensor:
        """Second derivative of the state vector (not calculated here)."""
        # For brevity, we'll return zeros as computing second derivatives is complex.
        return torch.zeros(4)

    @property
    def min_dt(self, oversampling_factor: float = 10.0) -> Tensor:
        """
        Calculate the minimum dt for good approximation based on the system's dynamics
        and the Nyquist criterion.

        Args:
            oversampling_factor (float): Factor by which to oversample the Nyquist rate (default is 10)

        Returns:
            Tensor: Minimum dt value
        """
        # Linearize the system around the upright position (theta ≈ 0)
        # The linearized natural frequency of the pendulum is omega_n = sqrt(g / L)

        omega_n = torch.sqrt(self.g / self.L)

        # Apply the Nyquist criterion with oversampling
        min_dt = (torch.pi) / (oversampling_factor * omega_n)

        # Ensure stability for the Euler integration method
        max_stable_dt = 2.0 / omega_n

        # Choose the smaller dt to satisfy both criteria
        min_dt = torch.min(min_dt, max_stable_dt)

        return min_dt
