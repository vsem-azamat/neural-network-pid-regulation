import tqdm
import torch
from torch import Tensor
import matplotlib.pyplot as plt

# Local imports
from classes.PID import PID
from classes.Systems import Trolley
from classes.Simulation import Simulation


class SimulationNN(Simulation):
	def __init__(self, time: Tensor, target: Tensor, disturbance: Tensor, dt: Tensor = torch.tensor(0.01)) -> None:
		super().__init__(time, target, disturbance, dt)
		self.model = None
		self.custom_loss = None
		self.optimizer = None

		self.feedback_Kp = torch.zeros(len(time))
		self.feedback_Ki = torch.zeros(len(time))
		self.feedback_Kd = torch.zeros(len(time))


	def run(self, trolley: Trolley, pid: PID) -> None:

		# Check if the model, loss function, and optimizer are defined
		if self.model is None or self.custom_loss is None or self.optimizer is None:
			raise Exception("Model, loss function, or optimizer is not defined")

		# This cycle like the one epoc in the training
		with tqdm.tqdm(total=len(self.time)-1) as pbar:
			pbar.colour = 'green'
			for i in range(1, len(self.time)):
				# Get data from the previous step
				target = self.target[i]
				disturbance = self.disturbance[i]
				position = trolley.get_position()

				# PID controller
				pid.KP = self.feedback_Kp[i-1]
				pid.KI = self.feedback_Ki[i-1]
				pid.KD = self.feedback_Kd[i-1]
				control_output = pid.compute(target, position, self.dt)
				self.feedback_U[i] = control_output

				# Update the trolley
				trolley.update(control_output, disturbance)

				# Get the current position
				new_position = trolley.get_position()
				self.feedback_X[i] = new_position

				# Reset gradient
				self.optimizer.zero_grad()

				# Build input vector X
				U = self.feedback_U[i]
				dUdKP = torch.abs((self.feedback_U[i] - self.feedback_U[i-1]) / (self.feedback_Kp[i] - self.feedback_Kp[i-1]))
				dUdKI = torch.abs((self.feedback_U[i] - self.feedback_U[i-1]) / (self.feedback_Ki[i] - self.feedback_Ki[i-1]))
				dUdKD = torch.abs((self.feedback_U[i] - self.feedback_U[i-1]) / (self.feedback_Kd[i] - self.feedback_Kd[i-1]))
		
				# Vector of X
				input_X = torch.tensor([U, dUdKP, dUdKI, dUdKD], requires_grad=True)

				# Predict the next coeeficients of the PID controller
				predicted_Y = self.model(input_X)
				Kp = predicted_Y[0].item()
				Ki = predicted_Y[1].item()
				Kd = predicted_Y[2].item()

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

				vector_K = torch.tensor([Kp_t, Kp_t_1, Ki_t, Ki_t_1, Kd_t, Kd_t_1], dtype=torch.float32, requires_grad=True)

				# Calculate the error
				E_t = target - new_position
				E_t_1 = self.feedback_E[i-1]
				self.feedback_E[i] = E_t
				vector_E = torch.tensor([E_t, E_t_1], dtype=torch.float32, requires_grad=True)
				
				# Calculate the loss
				loss = self.custom_loss(vector_K, vector_E, self.dt)

				# Backpropagation
				loss.backward()

				# Update the weights
				self.optimizer.step()

				# Update the progress bar
				pbar.update(1)
				pbar.set_description(f"Loss: {loss.item():.4f} | Error: {self.feedback_E[i]:.4f}, X_targ: {target:.4f} | U: {U:.4f} | Kp: {Kp:.4f}, Ki: {Ki:.4f}, Kd: {Kd:.4f}")


	def plot_K(self) -> None:

		# Convert to numpy
		time = self.time.cpu().detach().numpy()
		feedback_Kp = self.feedback_Kp.cpu().detach().numpy()
		feedback_Ki = self.feedback_Ki.cpu().detach().numpy()
		feedback_Kd = self.feedback_Kd.cpu().detach().numpy()

		print(self.time)
		print(self.feedback_Kp)
		print(self.feedback_Ki)
		print(self.feedback_Kd)


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
		self.feedback_X = torch.zeros(len(self.time))
		self.feedback_U = torch.zeros(len(self.time))
		self.feedback_Kp = torch.zeros(len(self.time))
		self.feedback_Ki = torch.zeros(len(self.time))
		self.feedback_Kd = torch.zeros(len(self.time))