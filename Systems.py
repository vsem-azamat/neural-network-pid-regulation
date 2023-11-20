import torch
from torch import Tensor

from abc import ABC, abstractmethod
from typing import Tuple


class BaseSystem(ABC):
	@abstractmethod
	def update(self, control_output: Tensor, distrubance: Tensor) -> None:
		"""
		Update the position and velocity of the object

		Args:
			control_output (float): control output applied to the object
			distrubance (float): distrubance applied to the object

		Returns:
			None
		"""
		pass

	@abstractmethod
	def get_position(self) -> Tensor:
		"""
		Get the position of the object

		Args:
			None

		Returns:
			float: position of the object
		"""
		pass


class Trolley(BaseSystem):
	def __init__(self, mass: Tensor, friction: Tensor, dt: Tensor) -> None:
		"""
		Initialize the trolley

		Args:
			mass (float): mass of the trolley
			friction (float): friction coefficient of the trolley
			dt (float): time step between the current and previous position

		Returns:
			None
		"""
		self.mass: Tensor = mass # kg
		self.friction: Tensor = friction # N*s/m
		self.spring_constant: Tensor = torch.tensor(50.) # N/m
		self.dt: Tensor = dt # s
		self.position: Tensor = torch.tensor(0.) # m
		self.delta_position: Tensor = torch.tensor(0.) # m
		self.velocity: Tensor = torch.tensor(0.) # m/s
		self.F: Tensor = torch.tensor(50.) # N


	def update(self, control_output: Tensor, distrubance: Tensor = 0) -> None:
		"""
		Update the position and velocity of the trolley
		
		Args:
			TODO: select best unit for the demonstration
		
		Returns:
			None
			
		Equation of model:
            F = ma
            a = F/m
            a = F/m - friction*v/m
            v = v + a*dt
            x = x + v*dt
		"""
		F = control_output
		acceleration = F / self.mass - self.friction * self.velocity / self.mass - self.spring_constant * self.delta_position / self.mass
		self.velocity += acceleration * self.dt
		position = self.position + self.velocity * self.dt
		self.delta_position = position - self.position
		self.position = position


	def get_position(self) -> Tensor:
		"""
		Get the position of the trolley

		Args:
			None

		Returns:
			float: position of the trolley
		"""
		return self.position
	

	def get_U(self) -> Tuple[Tensor, Tensor, Tensor]:
		return self.position, self.position/self.dt, self.position/(self.dt**2)


class ContinuousTankHeating(BaseSystem):
	def __init__(self, dt: Tensor) -> None:
		self.dt: Tensor = dt
		self.Tf: Tensor = torch.tensor(300.)
		self.T: Tensor = torch.tensor(300.)
		self.epsilon: Tensor = torch.tensor(1.)
		self.tau: Tensor = torch.tensor(4.)
		self.Q: Tensor = torch.tensor(2.)

	# TODO: fix the equation of the model
	def update(self, control_output: Tensor, distrubance: Tensor = torch.tensor(0.)) -> None:
		"""
		Update the position and velocity of the trolley
		
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

		dTdt = 1/(1 + self.epsilon) * (1/self.tau * (self.Tf - self.T) + self.Q * (Tq - self.T))
		self.T += dTdt * self.dt

	def get_position(self) -> Tensor:
		return self.T