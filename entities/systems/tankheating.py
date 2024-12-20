import torch
from torch import Tensor

from .base import BaseSystem


class TankHeating(BaseSystem):
    def __init__(self, dt: Tensor) -> None:
        self.dt: Tensor = dt
        self.Tf: Tensor = torch.tensor(300.0)
        self.T: Tensor = torch.tensor(300.0)
        self.epsilon: Tensor = torch.tensor(1.0)
        self.tau: Tensor = torch.tensor(4.0)
        self.Q: Tensor = torch.tensor(2.0)

    def apply_control(
        self, control_output: Tensor, disturbance: Tensor = torch.tensor(0.0)
    ) -> Tensor:
        """
        Update the temperature of the tank based on the control output

        Equation of model:
                dTdt = 1/(1+epsilon) * [1/tau * (Tf - T) + Q * (Tq - T)]

        Vars:
                Tq: target temperature
                Tf: temperature of the incoming fluid
                T: current temperature
                tau: residence time
                epsilon: ratio of the heat capacity of the tank to the heat capacity of the fluid
        """
        Tq = control_output

        dTdt = (
            1
            / (1 + self.epsilon)
            * (1 / self.tau * (self.Tf - self.T) + self.Q * (Tq - self.T))
        )
        self.T += dTdt * self.dt
        return self.T

    def reset(self) -> None:
        """Reset the system state to initial conditions."""
        self.T = torch.tensor(300.0)

    @property
    def X(self) -> Tensor:
        """Return the current temperature."""
        return self.T

    @property
    def dXdT(self) -> Tensor:
        """Return the rate of change of temperature."""
        Tq = self.T  # Assuming Tq is the target temperature
        dTdt = (
            1
            / (1 + self.epsilon)
            * (1 / self.tau * (self.Tf - self.T) + self.Q * (Tq - self.T))
        )
        return dTdt

    @property
    def d2XdT2(self) -> Tensor:
        """Return the second derivative of temperature (not applicable for this system)."""
        return torch.tensor(0.0)

    @property
    def min_dt(self, oversampling_factor: float = 10.0) -> Tensor:
        """
        Return a fixed minimum dt for the tank heating system.

        Args:
            oversampling_factor (float): Factor by which to oversample the Nyquist rate (default is 10)

        Returns:
            Tensor: Minimum dt value
        """
        return self.dt
