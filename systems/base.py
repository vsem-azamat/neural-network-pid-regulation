from abc import ABC, abstractmethod
from torch import Tensor


class BaseSystem(ABC):
	@abstractmethod
	def apply_control(self, control_output: Tensor, distrubance: Tensor) -> None:
		"""
		Update the position and velocity of the object

		Args:
			control_output (float): control output applied to the object
			distrubance (float): distrubance applied to the object

		Returns:
			None
		"""
		pass
