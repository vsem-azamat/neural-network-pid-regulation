import torch
from torch import Tensor
import matplotlib.pyplot as plt

# Local imports
from classes.pid import PID
from systems.base import BaseSystem


class Simulation:
	def __init__(self, time: Tensor, target: Tensor, disturbance: Tensor, dt: Tensor = torch.tensor(0.01)) -> None:
		"""
		Initialize the simulation environment

		Args:
			time (Tensor): time array of the simulation
			target (Tensor): array of target values
			disturbance (Tensor): array of disturbance values
			dt (Tensor): time step between the current and previous position

		Returns:
			None
		"""
		self.time: Tensor = time
		self.target: Tensor = target
		self.disturbance: Tensor = disturbance
		self.dt: Tensor = dt

		self.feedback_Y = torch.zeros(len(self.time))
		self.feedback_U = torch.zeros(len(self.time))
		self.feedback_E = torch.zeros(len(self.time))


	def run(self, simulationObj: BaseSystem, pid: PID) -> None:
		"""
		Run the simulation

		Args:
			simulationObj (BaseSimulationObject): simulation object
			pid (PID): PID controller

		Returns:
			None
		"""
		for i in range(1, len(self.time)):
			target = self.target[i]
			disturbance = self.disturbance[i]
			position = simulationObj.get_position()

			# Compute the control output
			control_output = pid.compute(target, position, self.dt)
			self.feedback_U[i] = control_output

			# Update the simulation object
			simulationObj.apply_control(control_output, disturbance)

			# Save to the feedback updated position
			new_position = simulationObj.get_position()
			self.feedback_Y[i] =  new_position

			# Compute and save the error
			error = target - new_position
			self.feedback_E[i] = error


	def plot(self) -> None:
		"""
		Plot the results of the simulation

		Args:
			None

		Returns:
			None
		"""
		time = self.time.cpu().detach().numpy()
		target = self.target.cpu().detach().numpy()
		disturbance = self.disturbance.cpu().detach().numpy()
		feedback_X = self.feedback_Y.cpu().detach().numpy()
		feedback_U = self.feedback_U.cpu().detach().numpy()

		plt.figure()
		plt.plot(time, target, label='Y_target', linestyle='--', color='red')
		plt.plot(time, disturbance, label='D_disturbance', color='black')
		plt.plot(time, feedback_X, label='Y_PID', color='orange')
		plt.plot(time, feedback_U, label='U_PID', color='blue')
		plt.title('PID Control')
		plt.xlabel('X_time')
		plt.ylabel('Value')
		plt.legend()
		plt.grid()
		plt.show()
