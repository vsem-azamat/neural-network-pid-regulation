import torch
from torch import Tensor

from systems.base import BaseSystem



class TankHeating(BaseSystem):
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