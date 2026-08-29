"""Lumped-capacity thermal plant: a first-order system with a heat input."""

import torch
from torch import Tensor

from .base import ZERO, BaseSystem

ROOM_TEMPERATURE = torch.tensor(293.15)  # 20 °C in kelvin


class Thermal(BaseSystem):
    """C·dT/dt = Q + d − h·(T − T_amb)

    Newton's law of cooling is written against the *ambient* temperature.  The
    earlier form, ``C·dT/dt = Q − h·T``, exchanges heat with 0 K instead: it
    needs 4 kW just to hold 400 K, and with the heater off it drives the plant
    towards absolute zero rather than towards room temperature.
    """

    def __init__(
        self,
        thermal_capacity: Tensor,
        heat_transfer_coefficient: Tensor,
        dt: Tensor,
        initial_temperature: Tensor = ROOM_TEMPERATURE,
        ambient_temperature: Tensor | None = None,
    ) -> None:
        """
        Args:
            thermal_capacity: Heat capacity C, J/K.
            heat_transfer_coefficient: Loss coefficient h, W/K.
            dt: Integration step, s.
            initial_temperature: Starting temperature, K.
            ambient_temperature: Environment temperature, K. Defaults to the
                initial temperature, i.e. the plant starts at equilibrium.
        """
        self.thermal_capacity = torch.as_tensor(thermal_capacity, dtype=torch.float32)
        self.heat_transfer_coefficient = torch.as_tensor(
            heat_transfer_coefficient, dtype=torch.float32
        )
        self.dt = torch.as_tensor(dt, dtype=torch.float32)
        self.initial_temperature = torch.as_tensor(
            initial_temperature, dtype=torch.float32
        )
        self.ambient_temperature = (
            self.initial_temperature
            if ambient_temperature is None
            else torch.as_tensor(ambient_temperature, dtype=torch.float32)
        )

        self.temperature = self.initial_temperature.clone()
        self.temp_derivative = torch.tensor(0.0)

    def apply_control(self, control_output: Tensor, disturbance: Tensor = ZERO) -> Tensor:
        """Advance one step under heat input ``control_output`` plus ``disturbance`` (W)."""
        heat_loss = self.heat_transfer_coefficient * (
            self.temperature - self.ambient_temperature
        )
        self.temp_derivative = (
            control_output + disturbance - heat_loss
        ) / self.thermal_capacity
        self.temperature = self.temperature + self.temp_derivative * self.dt
        return self.temperature

    def reset(self) -> None:
        self.temperature = self.initial_temperature.clone()
        self.temp_derivative = torch.tensor(0.0)

    def detach_state(self) -> None:
        self.temperature = self.temperature.detach()
        self.temp_derivative = self.temp_derivative.detach()

    @property
    def X(self) -> Tensor:
        return self.temperature

    @property
    def dXdT(self) -> Tensor:
        return self.temp_derivative

    @property
    def d2XdT2(self) -> Tensor:
        return torch.tensor(0.0)

    def min_dt(self, oversampling_factor: float = 10.0) -> Tensor:
        """Sampling step that resolves the plant's time constant τ = C/h."""
        tau = self.thermal_capacity / self.heat_transfer_coefficient
        nyquist_dt = torch.pi * tau / oversampling_factor
        max_stable_dt = 2.0 * tau
        return torch.min(nyquist_dt, max_stable_dt)

    @property
    def tau(self) -> Tensor:
        """Open-loop time constant, s."""
        return self.thermal_capacity / self.heat_transfer_coefficient

    @property
    def steady_state_power(self) -> Tensor:
        """Heat input needed to hold the current temperature, W."""
        return self.heat_transfer_coefficient * (
            self.temperature - self.ambient_temperature
        )
