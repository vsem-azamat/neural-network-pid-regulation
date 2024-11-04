import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Generic, TypeVar, List

T = TypeVar("T", torch.Tensor, np.ndarray)


@dataclass
class LearningConfig:
    dt: torch.Tensor
    num_epochs: int
    train_time: float
    learning_rate: float

    @property
    def train_steps(self) -> int:
        return int(self.train_time / self.dt)


@dataclass
class SimulationConfig(Generic[T]):
    """
    Configuration for simulation run

    setpoints: List[T] - list of setpoints for the simulation
    dt: T - time step Delta t
    sequence_length: int - length of the sequence for LSTM
    sequence_step: int - step for the sequence for LSTM
    pid_gain_factor: int - multiplier for PID gains
    """

    setpoints: List[T]
    dt: T
    sequence_length: int = 100
    sequence_step: int = 5
    pid_gain_factor: int = 100


@dataclass
class SimulationResults(Generic[T]):
    time_points: List[T] = field(default_factory=list)
    positions: List[T] = field(default_factory=list)
    control_outputs: List[T] = field(default_factory=list)
    rbf_predictions: List[T] = field(default_factory=list)
    error_history: List[T] = field(default_factory=list)
    error_diff_history: List[T] = field(default_factory=list)
    kp_values: List[T] = field(default_factory=list)
    ki_values: List[T] = field(default_factory=list)
    kd_values: List[T] = field(default_factory=list)
    pid_params: List[T] = field(default_factory=list)
    angle_history: List[T] = field(default_factory=list)
    losses: List[T] = field(default_factory=list)
    setpoints: List[T] = field(default_factory=list)  # Add this line

    @classmethod
    def with_length(cls, length: int) -> "SimulationResults[torch.Tensor]":
        return SimulationResults(
            time_points=[torch.tensor(0.0) for _ in range(length)],
            positions=[torch.tensor(0.0) for _ in range(length)],
            control_outputs=[torch.tensor(0.0) for _ in range(length)],
            rbf_predictions=[torch.tensor(0.0) for _ in range(length)],
            error_history=[torch.tensor(0.0) for _ in range(length)],
            error_diff_history=[torch.tensor(0.0) for _ in range(length)],
            kp_values=[torch.tensor(0.0) for _ in range(length)],
            ki_values=[torch.tensor(0.0) for _ in range(length)],
            kd_values=[torch.tensor(0.0) for _ in range(length)],
            pid_params=[torch.tensor(0.0) for _ in range(length)],
            angle_history=[torch.tensor(0.0) for _ in range(length)],
            losses=[torch.tensor(0.0) for _ in range(length)],
            setpoints=[torch.tensor(0.0) for _ in range(length)],
        )

    def detach_tensors(self) -> "SimulationResults[torch.Tensor]":
        results = self.clone_tensors()
        for tensor_list in results.__dict__.values():
            for tensor in tensor_list:
                tensor.detach_()
        return results

    def clone_tensors(self) -> "SimulationResults[torch.Tensor]":
        for key, value in self.__dict__.items():
            setattr(self, key, [tensor.clone() for tensor in value])
        return self

    def to_numpy(self) -> "SimulationResults[np.ndarray]":
        results = self.detach_tensors()
        for tensor_list in results.__dict__.values():
            for i, tensor in enumerate(tensor_list):
                tensor_list[i] = tensor.numpy()
        return results
