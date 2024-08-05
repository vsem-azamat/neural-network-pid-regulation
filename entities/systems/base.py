from abc import ABC, abstractmethod
from torch import Tensor


class BaseSystem(ABC):
	@abstractmethod
	def apply_control(self, control_output: Tensor, distrubance: Tensor | None = None) -> None:
		"""
		Update the position and velocity of the object

		Args:
			control_output (float): control output applied to the object
			distrubance (float): distrubance applied to the object

		Returns:
			None
		"""
		pass

	@property
	@abstractmethod
	def X(self) -> Tensor:
		"""
		Return the position of the object

		Returns:
			float: position of the object
		"""
		pass

	@property
	@abstractmethod
	def dXdT(self) -> Tensor:
		"""
		Return the velocity of the object

		Returns:
			float: velocity of the object
		"""
		pass

	@property
	@abstractmethod
	def d2XdT2(self) -> Tensor:
		"""
		Return the acceleration of the object

		Returns:
			float: acceleration of the object
		"""
		pass
