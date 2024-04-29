import torch
from torch import Tensor


class PID:
	def __init__(self, initial_KP: Tensor, initial_KI: Tensor, initial_KD: Tensor):
		self.Kp = initial_KP
		self.Ki = initial_KI
		self.Kd = initial_KD

        # PID states
		self.error = torch.tensor(0.)
		self.error_last = torch.tensor(0.)
		self.integral_error = torch.tensor(0.)
		self.derivative_error = torch.tensor(0.)

		# PID saturation limits
		self.saturation_max = torch.tensor(500.)
		self.saturation_min = torch.tensor(-500.)

		# PID
	

	# Function to update PID gains based on external values
	def update_gains(self, new_Kp: Tensor, new_Ki: Tensor, new_Kd: Tensor) -> None:
		self.Kp = new_Kp
		self.Ki = new_Ki
		self.Kd = new_Kd


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
		self.error = target - position
		self.integral_error += self.error * dt
		self.derivative_error = (self.error - self.error_last) / dt #find the derivative of the error (how the error changes with time)
		self.error_last = self.error # update the error
		
		# Calculate the output
		output = \
			self.Kp * self.error.detach() + \
			self.Ki * self.integral_error.detach() + \
			self.Kd * self.derivative_error.detach()

		# return output
		return torch.clamp(output, self.saturation_min, self.saturation_max)
  
