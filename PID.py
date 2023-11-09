from Systems import *


class PID:
	"""PID controller class"""
	def __init__(self, KP: float, KI: float, KD: float) -> None:
		"""
		A PID controller is characterized by three parameters: proportional gain (KP), integral gain (KI), and derivative gain (KD).
		
		Args:
            KP (float): proportional gain
            KI (float): integral gain
            KD (float): derivative gain
		"""
		# PID parameters
		self.kp = KP
		self.ki = KI
		self.kd = KD	

        # PID states
		self.error_last = 0
		self.integral_error = 0

		# PID saturation limits
		self.saturation_max: float = 1
		self.saturation_min: float = -1
	
    
	def compute(self, target: float, position: float, dt: float) -> float:
		"""
		Calculate the output of the PID controller
		
		Args:
            target (float): target value
			position (float): current value
			dt (float): time step between the current and previous position
			
        Returns:
            float: output of the PID controller
		"""
		error = target - position # error = target - current
		derivative_error = (error - self.error_last) / dt #find the derivative of the error (how the error changes with time)
		self.integral_error += error * dt #error build up over time
		
		output = self.kp*error + self.ki*self.integral_error + self.kd*derivative_error # compute the output
		self.error_last = error # update the error

		if output > self.saturation_max and self.saturation_max:
			output = self.saturation_max
		elif output < self.saturation_min and self.saturation_min:
			output = self.saturation_min
		return output


	def setLims(self, min: float, max: float) -> None:
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

