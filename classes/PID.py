import torch
from torch import Tensor

from typing import Optional

class PID:
	"""PID controller class"""
	def __init__(self, KP: Tensor, KI: Tensor, KD: Tensor) -> None:
		"""
		A PID controller is characterized by three parameters: proportional gain (KP), integral gain (KI), and derivative gain (KD).
		
		Args:
            KP (float): proportional gain
            KI (float): integral gain
            KD (float): derivative gain
		"""

		# PID parameters
		self.KP: Tensor = torch.tensor(KP)
		self.KI: Tensor = torch.tensor(KI)
		self.KD: Tensor = torch.tensor(KD)

        # PID states
		self.error: Tensor = torch.tensor(0.)
		self.error_last: Tensor = torch.tensor(0.)
		self.integral_error: Tensor = torch.tensor(0.)
		self.derivative_error: Tensor = torch.tensor(0.)

		# PID saturation limits
		self.saturation_max: Optional[Tensor] = torch.tensor(20.)
		self.saturation_min: Optional[Tensor] = torch.tensor(-20.)
	
    
	def compute(self, target: Tensor, position: Tensor, dt: Tensor) -> Tensor:
		"""
		Calculate the output of the PID controller
		
		Args:
            target (float): target value
			position (float): current value
			dt (float): time step between the current and previous position
			
        Returns:
            float: output of the PID controller
		"""
		# Calculate the errors
		self.error = target - position # error = target - current
		self.integral_error += self.error * dt #error build up over time
		self.derivative_error = (self.error - self.error_last) / dt #find the derivative of the error (how the error changes with time)
		self.error_last = self.error # update the error
		
		# Calculate the output
		output = \
			self.KP * self.error + \
			self.KI * self.integral_error + \
			self.KD * self.derivative_error

		if self.saturation_max and output > self.saturation_max:
			output = self.saturation_max
		elif self.saturation_min and output < self.saturation_min:
			output = self.saturation_min
		return output


	def setLims(self, min: Optional[Tensor], max: Optional[Tensor]) -> None:
		"""
		Set the saturation limits for the PID controller
		
		Args:
            min (float): minimum value of the output
            max (float): maximum value of the output
			
        Returns:
            None
		"""
		self.saturation_max = max
		self.saturation_min = min

