import torch
from torch import Tensor
from abc import ABC, abstractmethod


class BaseSystem(ABC):
    @abstractmethod
    def apply_control(
        self, control_output: Tensor, distrubance: Tensor = torch.tensor(0.0)
    ) -> Tensor:
        """
        Update the position and velocity of the object

        Args:
                control_output (float): control output applied to the object
                distrubance (float): distrubance applied to the object

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
