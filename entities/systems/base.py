import torch
import numpy as np
from torch import Tensor
from abc import ABC, abstractmethod
from typing import Literal


class BaseSystem(ABC):
    @abstractmethod
    def apply_control(
        self, control_output: Tensor, disturbance: Tensor = torch.tensor(0.0)
    ) -> Tensor:
        """
        Update the position and velocity of the object

        Args:
                control_output (float): control output applied to the object
                disturbance (float): distrubance applied to the object

        Returns:
                None
        """
        raise NotImplementedError("apply_control not implemented")

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the object to its initial state

        Returns:
                None
        """
        raise NotImplementedError("reset not implemented")

    @property
    @abstractmethod
    def min_dt(self, oversampling_factor: float = 10.0) -> Tensor:
        """
        Return the minimum time step

        Returns:
                Tensor: minimum time step
        """
        raise NotImplementedError("min_dt not implemented")

    @property
    @abstractmethod
    def X(self) -> Tensor:
        """
        Return the position of the object

        Returns:
                float: position of the object
        """
        raise NotImplementedError("X not implemented")

    @property
    @abstractmethod
    def dXdT(self) -> Tensor:
        """
        Return the velocity of the object

        Returns:
                float: velocity of the object
        """
        raise NotImplementedError("dXdT not implemented")

    @property
    @abstractmethod
    def d2XdT2(self) -> Tensor:
        """
        Return the acceleration of the object

        Returns:
                float: acceleration of the object
        """
        raise NotImplementedError("d2XdT2 not implemented")

    def __estimate_process_parameters(
        self,
        dt,
        steps,
        initial_input: float,
        final_input: float,
    ) -> tuple[float, float, float]:
        time_data = np.linspace(0, dt * steps, steps)
        self.reset()

        output_data = []
        for i, t in enumerate(time_data):
            self.apply_control(torch.tensor(final_input))
            output_data.append(self.X.item())

        output_data = np.array(output_data)

        delta_u = final_input - initial_input
        if delta_u == 0:
            raise ValueError("No step change in input detected.")

        y0 = output_data[0]
        y_inf = output_data[-1]
        delta_y = y_inf - y0
        if delta_y == 0:
            raise ValueError("System output does not respond to input.")

        K = delta_y / delta_u

        y_35 = y0 + 0.35 * delta_y
        y_85 = y0 + 0.85 * delta_y

        def find_time_for_level(level):
            idx = np.where(output_data >= level)[0]
            if len(idx) == 0:
                raise ValueError(f"Output never reaches the required level ({level}).")
            return time_data[idx[0]]

        try:
            t_35 = find_time_for_level(y_35)
            t_85 = find_time_for_level(y_85)
        except ValueError as e:
            raise ValueError(f"Error finding time for level: {e}")

        delta_t = t_85 - t_35
        T_est = 1.5 * delta_t
        L_est = t_35 - 0.29 * T_est
        if L_est < 0:
            L_est = 0.0

        return K, L_est, T_est

    def tune_pid(
        self,
        dt,
        steps,
        method: Literal["ziegler_nichols", "cohen_coon", "pid_imc"],
        initial_input=0.0,
        final_input=3000.0,
        **kwargs,
    ) -> tuple[float, float, float]:
        K, L, T = self.__estimate_process_parameters(
            dt, steps, initial_input, final_input
        )
        if method == "ziegler_nichols":
            return self._tune_ziegler_nichols(K, L, T)
        elif method == "cohen_coon":
            return self._tune_cohen_coon(K, L, T)
        elif method == "pid_imc":
            lambda_value = kwargs.get("lambda_value", 1.0)
            return self._tune_pid_imc(K, L, T, lambda_value)
        else:
            raise ValueError(f"Unknown tuning method: {method}")

    def _tune_ziegler_nichols(
        self, K: float, L: float, T: float
    ) -> tuple[float, float, float]:
        """
        Tune PID controller using Ziegler-Nichols method.

        Args:
            K (float): Process gain
            L (float): Process dead time
            T (float): Process time constant

        Returns:
            tuple[float, float, float]: PID parameters (Kp, Ki, Kd)

        Mathematical Description:
            Kp = 1.2 * (T / (L * K))
            Ki = Kp / (2.0 * L)
            Kd = 0.5 * L

        Explanation:
            - Kp is proportional to the ratio of T to L and K.
            - Ki is derived from Kp and L.
            - Kd is proportional to L.
        """
        if L == 0 or K == 0:
            raise ValueError(
                "L or K is zero, cannot compute Ziegler-Nichols parameters."
            )
        Kp = 1.2 * (T / (L * K))
        Ki = Kp / (2.0 * L)
        Kd = 0.5 * L
        return Kp, Ki, Kd

    def _tune_cohen_coon(
        self, K: float, L: float, T: float
    ) -> tuple[float, float, float]:
        """
        Tune PID controller using Cohen-Coon method.

        Args:
            K (float): Process gain
            L (float): Process dead time
            T (float): Process time constant

        Returns:
            tuple[float, float, float]: PID parameters (Kp, Ki, Kd)

        Mathematical Description:
            ratio = T / L
            Kp = (1 / K) * ((1.35 * ratio + 0.27) / (1 + 0.6 * ratio))
            Ti = T * ((2.5 * (L / T) + 0.9) / (1 + 0.6 * (L / T)))
            Td = 0.37 * L * (T / (T + 0.2 * L))
            Ki = Kp / Ti
            Kd = Td

        Explanation:
            - Kp is adjusted based on the ratio of T to L.
            - Ti and Td are derived from T and L.
            - Ki is derived from Kp and Ti.
        """
        ratio = T / L
        Kp = (1 / K) * ((1.35 * ratio + 0.27) / (1 + 0.6 * ratio))
        Ti = T * ((2.5 * (L / T) + 0.9) / (1 + 0.6 * (L / T)))
        Td = 0.37 * L * (T / (T + 0.2 * L))
        Ki = Kp / Ti
        Kd = Td
        return Kp, Ki, Kd

    def _tune_pid_imc(
        self, K: float, L: float, T: float, lambda_value=1.0
    ) -> tuple[float, float, float]:
        """
        Tune PID controller using IMC method.

        Args:
            K (float): Process gain
            L (float): Process dead time
            T (float): Process time constant
            lambda_value (float): Desired closed-loop time constant (default is 1.0)

        Returns:
            tuple[float, float, float]: PID parameters (Kp, Ki, Kd)

        Mathematical Description:
            Kp = T / (K * (L + lambda_value))
            Ki = Kp / (T + L)
            Kd = Kp * (L / (T + L))

        Explanation:
            - Kp is inversely proportional to K and the sum of L and lambda_value.
            - Ki is derived from Kp and the sum of T and L.
            - Kd is proportional to L and inversely proportional to the sum of T and L.
        """
        Kp = T / (K * (L + lambda_value))
        Ki = Kp / (T + L)
        Kd = Kp * (L / (T + L))
        return Kp, Ki, Kd
