import torch
from torch import Tensor

from .base import BaseSystem


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
		self.mass = mass
		self.friction = friction
		self.spring_constant = torch.tensor(5.)
		self.dt = dt

		self.position = torch.tensor(0.)
		self.delta_position = torch.tensor(0.)
		self.velocity = torch.tensor(0.)
		self.F = torch.tensor(50.)


	def apply_control(self, control_output: Tensor, distrubance: Tensor = torch.tensor(0.)) -> Tensor:
		"""
		Update the position and velocity of the trolley based on the control output
		
		Args:
			TODO: select best unit for the demonstration
		
		Returns:
			None
		
		Equation of model:
            F = ma
            a = F/m
            a = F/m - friction*v/m - spring_constant*x/m
            v = v + a*dt
            x = x + v*dt
		"""
		assert control_output is not None, "Control output is None"
		F = control_output
		acceleration = F / self.mass - self.friction * self.velocity.detach() / self.mass - self.spring_constant * self.position.detach() / self.mass
		self.velocity = self.velocity.detach() + acceleration * self.dt
		self.position = self.position.detach() + self.velocity * self.dt
		return self.position


	def get_position(self) -> Tensor:
		"""
		Get the position of the trolley

		Args:
			None

		Returns:
			float: position of the trolley
		"""
		return self.position
	

	def get_U(self) -> tuple[Tensor, Tensor, Tensor]:
		return self.position, self.position/self.dt, self.position/(self.dt**2)
	

	def reset(self) -> None:
		"""
		Reset the position and velocity of the trolley

		Args:
			None

		Returns:
			None
		"""
		self.position = torch.tensor(0, dtype=torch.float32)
		self.velocity = torch.tensor(0, dtype=torch.float32)
		self.delta_position = torch.tensor(0, dtype=torch.float32)


