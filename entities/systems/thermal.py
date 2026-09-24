"""Lumped-capacity thermal plant: a first-order system with a heat input."""

import torch
from torch import Tensor

from .base import ZERO, BaseSystem

ROOM_TEMPERATURE = torch.tensor(293.15)  # 20 °C in kelvin


class Thermal(BaseSystem):
    """C·dT/dt = Q + d − h·(T − T_amb) − σ·(T⁴ − T_amb⁴)

    Newton's law of cooling is written against the *ambient* temperature.  The
    earlier form, ``C·dT/dt = Q − h·T``, exchanges heat with 0 K instead: it
    needs 4 kW just to hold 400 K, and with the heater off it drives the plant
    towards absolute zero rather than towards room temperature.

    The optional radiative term (Stefan–Boltzmann, ``radiative_coefficient``)
    defaults to zero, which recovers the linear plant. Switched on, it is what
    makes this plant a genuine gain-scheduling problem: radiation contributes an
    effective loss coefficient of 4σT³, so the plant's time constant shrinks
    sharply as it heats up. Between 300 K and 500 K the total loss coefficient
    changes several-fold, and a controller tuned at one end is badly detuned at
    the other. This is the textbook industrial reason to schedule gains on
    temperature.
    """

    def __init__(
        self,
        thermal_capacity: Tensor,
        heat_transfer_coefficient: Tensor,
        dt: Tensor,
        initial_temperature: Tensor = ROOM_TEMPERATURE,
        ambient_temperature: Tensor | None = None,
        radiative_coefficient: Tensor | float = 0.0,
    ) -> None:
        """
        Args:
            thermal_capacity: Heat capacity C, J/K.
            heat_transfer_coefficient: Loss coefficient h, W/K.
            dt: Integration step, s.
            initial_temperature: Starting temperature, K.
            ambient_temperature: Environment temperature, K. Defaults to the
                initial temperature, i.e. the plant starts at equilibrium.
            radiative_coefficient: Effective εσA for the radiative loss, W/K⁴.
                Zero gives a purely linear plant.
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

        self.radiative_coefficient = torch.as_tensor(
            radiative_coefficient, dtype=torch.float32
        )

        self.temperature = self.initial_temperature.clone()
        self.temp_derivative = torch.tensor(0.0)

    def apply_control(
        self, control_output: Tensor, disturbance: Tensor = ZERO
    ) -> Tensor:
        """Advance one step under heat input plus ``disturbance``, both in W."""
        heat_loss = self.heat_transfer_coefficient * (
            self.temperature - self.ambient_temperature
        )
        if float(self.radiative_coefficient) != 0.0:
            heat_loss = heat_loss + self.radiative_coefficient * (
                self.temperature**4 - self.ambient_temperature**4
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

    def min_dt(
        self, oversampling_factor: float = 10.0, amplitude: float = 0.0
    ) -> Tensor:
        """Sampling step that resolves the plant's time constant τ = C/h_eff.

        ``amplitude`` is read as an operating temperature when it is non-zero,
        because the radiative plant's time constant depends on it.
        """
        if amplitude:
            tau = self.thermal_capacity / self.effective_loss_coefficient(amplitude)
            return torch.min(
                torch.pi * tau / oversampling_factor, 2.0 * tau
            )
        tau = self.thermal_capacity / self.effective_loss_coefficient()
        nyquist_dt = torch.pi * tau / oversampling_factor
        max_stable_dt = 2.0 * tau
        return torch.min(nyquist_dt, max_stable_dt)

    def effective_loss_coefficient(self, temperature: float | None = None) -> Tensor:
        """dQ_loss/dT at a given temperature: h + 4σT³.

        This is the plant's local gain, and on the radiative plant it is a strong
        function of where the plant is operating.
        """
        T = self.temperature if temperature is None else torch.tensor(float(temperature))
        return self.heat_transfer_coefficient + 4.0 * self.radiative_coefficient * T**3

    @property
    def tau(self) -> Tensor:
        """Open-loop time constant at the current temperature, s."""
        return self.thermal_capacity / self.effective_loss_coefficient()

    @property
    def is_nonlinear(self) -> bool:
        return float(self.radiative_coefficient) != 0.0

    @property
    def steady_state_power(self) -> Tensor:
        """Heat input needed to hold the current temperature, W."""
        return self.heat_transfer_coefficient * (
            self.temperature - self.ambient_temperature
        )
