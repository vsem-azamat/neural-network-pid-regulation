import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt

# Local imports
from PID import PID
from SimObjects import *


class Simulation:
	def __init__(self, time: NDArray, target: NDArray, disturbance: NDArray, dt: float = 0.01) -> None:
		"""
		Initialize the simulation environment

		Args:
			time (numpy.ndarray): time array of the simulation
			target (numpy.ndarray): array of target values
			disturbance (numpy.ndarray): array of disturbance values
			dt (float): time step between the current and previous position

		Returns:
			None
		"""
		self.time = time
		self.target = target
		self.disturbance = disturbance
		self.dt = dt

		self.feedback = np.zeros(len(self.time))


	def run(self, simulationObj: BaseSimulationObject, pid: PID) -> None:
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

			# Update the simulation object
			simulationObj.update(control_output, disturbance)

			# Save the feedback
			self.feedback[i] = simulationObj.get_position()


	def plot(self) -> None:
		"""
		Plot the results of the simulation

		Args:
			None

		Returns:
			None
		"""
		plt.figure()
		plt.plot(self.time, self.disturbance, label='Disturbance')
		plt.plot(self.time, self.feedback, label='Feedback')
		plt.plot(self.time, self.target, label='Target', linestyle='--')
		plt.title('PID Control')
		plt.xlabel('Time')
		plt.ylabel('Value')
		plt.legend()
		plt.grid()
		plt.show()



if __name__ == "__main__":
	dt = 0.01

	time = np.arange(0, 100, dt)
	target = np.ones(len(time))*0.5
	disturbance = np.zeros(len(time))
	# disturbance[4000:4100] = 0.5

	pid = PID(KP=1, KI=0.1, KD=0.1)
	tr = Trolley(mass=1, friction=0.1, dt=dt)
	simulation = Simulation(time, target=target, disturbance=disturbance, dt=dt)
	simulation.run(tr, pid)
	simulation.plot()
