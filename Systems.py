from abc import ABC, abstractmethod


class BaseSystem(ABC):
	@abstractmethod
	def update(self, control_output: float, distrubance: float) -> None:
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
	def get_position(self) -> float:
		"""
		Get the position of the object

		Args:
			None

		Returns:
			float: position of the object
		"""
		pass


class Trolley(BaseSystem):
	def __init__(self, mass: float, friction: float, dt: float) -> None:
		"""
		Initialize the trolley

		Args:
			mass (float): mass of the trolley
			friction (float): friction coefficient of the trolley
			dt (float): time step between the current and previous position

		Returns:
			None
		"""
		self.mass: float = mass
		self.friction: float = friction
		self.dt: float = dt
		self.position: float = 0
		self.velocity: float = 0


	def update(self, force: float, distrubance: float) -> None:
		"""
		Update the position and velocity of the trolley
		
		Args:
			force (float): force applied to the trolley

		Returns:
			None
			
		Equation of model:
            F = ma
            a = F/m
            a = F/m - friction*v/m
            v = v + a*dt
            x = x + v*dt
		"""
		acceleration = force / self.mass - self.friction * self.velocity / self.mass - distrubance / self.mass
		self.velocity += acceleration * self.dt
		self.position += self.velocity * self.dt


	def get_position(self) -> float:
		"""
		Get the position of the trolley

		Args:
			None

		Returns:
			float: position of the trolley
		"""
		return self.position
