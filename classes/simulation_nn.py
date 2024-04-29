# Pip imports
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from tqdm import tqdm_notebook

# Local imports
from classes.pid import PID
from systems.trolley import BaseSystem
from classes.simulation import Simulation


class SimulationNN(Simulation):
	def __init__(self, time: Tensor, target: Tensor, disturbance: Tensor, dt: Tensor = torch.tensor(0.01)) -> None:
		super().__init__(time, target, disturbance, dt)
		self.model: Module = None
		self.custom_loss: Module = None
		self.optimizer: Optimizer = None

		self.system: BaseSystem = None
		self.pid: PID = None

		self.current_idx: int = 0

		self.feedback_Loss = torch.zeros(len(time))
		self.feedback_Kp = torch.ones(len(time))
		self.feedback_Ki = torch.ones(len(time))
		self.feedback_Kd = torch.ones(len(time))


	def __check(self) -> None:
		"""
		Check if the model, loss function, and optimizer are defined
		"""
		maping_inits = {
			"model": self.model,
			"custom_loss": self.custom_loss,
			"optimizer": self.optimizer,
			"system": self.system,
			"pid": self.pid
		}
		for key, value in maping_inits.items():
			if value is None:
				raise Exception(f"{key} is not defined")


	def run(self) -> None:
		"""
		Run the simulation
		"""

		# Check if the model, loss function, and optimizer are defined
		self.__check()

		# This cycle like the one epoc in the training
		# with tqdm.tqdm(total=len(self.time)-1) as pbar:
		with tqdm_notebook(total=len(self.time)-1) as pbar:
			pbar.colour = 'green'
			self.model.train()
			for i in range(1, len(self.time)):
				# Get data from the previous step
				target = self.target[i]
				disturbance = self.disturbance[i]
				position = self.system.get_position()

				# PID controller
				self.pid.Kp = self.feedback_Kp[i-1]
				self.pid.Ki = self.feedback_Ki[i-1]
				self.pid.Kd = self.feedback_Kd[i-1]
				U = self.pid.compute(target, position, self.dt)
				self.feedback_U[i] = U

				# Update the trolley
				self.system.apply_control(U, disturbance)

				# Get the current position
				Y_new = self.system.get_position()
				self.feedback_Y[i] = Y_new

				# ------------------- Build input X for model ------------------- #
				predicted_Y = self.model(input_X)	

				# Get the coefficients
				Kp = predicted_Y[0]
				Ki = predicted_Y[1]
				Kd = predicted_Y[2]

				# Save the coefficients
				self.feedback_Kp[i] = Kp
				self.feedback_Ki[i] = Ki
				self.feedback_Kd[i] = Kd

				# Vector of K coefficients
				Kp_t = self.feedback_Kp[i]
				Kp_t_1 = self.feedback_Kp[i-1]
				Ki_t = self.feedback_Ki[i]
				Ki_t_1 = self.feedback_Ki[i-1]
				Kd_t = self.feedback_Kd[i]
				Kd_t_1 = self.feedback_Kd[i-1]

				vector_K = torch.tensor([Kp_t, Kp_t_1, Ki_t, Ki_t_1, Kd_t, Kd_t_1], requires_grad=True)

				# Calculate the error
				E_t = target - Y_new
				E_t_1 = self.feedback_E[i-1]
				self.feedback_E[i] = E_t
				vector_E = torch.tensor([E_t, E_t_1], requires_grad=True)
				
				# Calculate the loss
				loss = self.custom_loss(vector_K, vector_E, self.dt)
				self.feedback_Loss[i] = loss
				
				# Backpropagation
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				# Update the progress bar
				pbar.update(1)
				# Set the description 
				pbar.set_description(f"L: {loss.item():.4f} | E: {self.feedback_E[i]:.4f}, X_: {target:.4f} | U: {U:.4f} | Kp: {Kp:.4f}, Ki: {Ki:.4f}, Kd: {Kd:.4f}")



	def __iter__(self) -> 'SimulationNN':
		return self
	
	
	def __next__(self) -> None:
		if self.current_idx >= len(self.time):
			raise StopIteration
		

		KP = self.feedback_Kp[self.current_idx]
		KI = self.feedback_Ki[self.current_idx]
		KD = self.feedback_Kd[self.current_idx]
		

	def plot_K(self) -> None:

		# Convert to numpy
		time = self.time.cpu().detach().numpy()
		feedback_Kp = self.feedback_Kp.cpu().detach().numpy()
		feedback_Ki = self.feedback_Ki.cpu().detach().numpy()
		feedback_Kd = self.feedback_Kd.cpu().detach().numpy()

		# Plot the coefficients
		fig, ax = plt.subplots(1, 3, figsize=(15, 5))
		ax[0].plot(time, feedback_Kp, label='Kp')
		ax[0].set_xlabel('Time [s]')
		ax[0].set_ylabel('Kp')
		ax[0].legend()
		ax[1].plot(time, feedback_Ki, label='Ki')
		ax[1].set_xlabel('Time [s]')
		ax[1].set_ylabel('Ki')
		ax[1].legend()
		ax[2].plot(time, feedback_Kd, label='Kd')
		ax[2].set_xlabel('Time [s]')
		ax[2].set_ylabel('Kd')
		ax[2].legend()
		ax[0].grid()
		ax[1].grid()
		ax[2].grid()
		plt.show()


	def reset(self) -> None:
		self.feedback_Y = torch.zeros(len(self.time))
		self.feedback_U = torch.zeros(len(self.time))
		# self.feedback_Kp = torch.zeros(len(self.time))
		# self.feedback_Ki = torch.zeros(len(self.time))
		# self.feedback_Kd = torch.zeros(len(self.time))

		self.current_idx = 0
