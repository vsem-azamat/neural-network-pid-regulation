import torch
from torch import Tensor


class PID:
	def __init__(self, initial_KP: Tensor, initial_KI: Tensor, initial_KD: Tensor):
		self.Kp = initial_KP
		self.Ki = initial_KI
		self.Kd = initial_KD

        # PID states
		self.e_k = torch.tensor(0.)
		self.e_k_1 = torch.tensor(0.)
		self.e_k_2 = torch.tensor(0.)
		self.u_k_1 = torch.tensor(0.)

		# PID saturation limits
		self.saturation_max = torch.tensor(10000.)
		self.saturation_min = torch.tensor(-10000.)


	def update_gains(self, new_Kp: Tensor, new_Ki: Tensor, new_Kd: Tensor) -> None:
		self.Kp = new_Kp
		self.Ki = new_Ki
		self.Kd = new_Kd


	def compute(self, error: Tensor, dt: Tensor) -> Tensor:
		# Store the errors
		self.e_k = error
		self.e_k_1 = self.e_k
		self.e_k_2 = self.e_k_1
		
		# Calculate the output
		u_k = \
			self.u_k_1 + \
			self.Kp * (self.e_k - self.e_k_1) + \
			self.Ki * self.e_k * dt + \
			self.Kd * ((self.e_k - self.e_k_1) - (self.e_k_1 - self.e_k_2)) / dt
		self.u_k_1 = u_k

		return torch.clamp(u_k, self.saturation_min, self.saturation_max)


	def set_limits(self, max_limit: Tensor, min_limit: Tensor) -> None:
		self.saturation_max = max_limit
		self.saturation_min = min_limit

