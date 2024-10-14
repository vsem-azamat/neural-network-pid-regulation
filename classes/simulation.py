import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Generic, TypeVar, List

T = TypeVar('T', torch.Tensor, np.ndarray)

@dataclass
class SimulationConfig(Generic[T]):
    setpoints: List[T]
    dt: T
    sequence_length: int = 100
    input_sequence_length: int = 10
    input_steps: int = 5

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
    def with_length(cls, length: int) -> 'SimulationResults[torch.Tensor]':
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
            setpoints=[torch.tensor(0.0) for _ in range(length)]
        )

    def detach_tensors(self) -> 'SimulationResults[torch.Tensor]':
        return SimulationResults(
            time_points=[time.clone().detach() for time in self.time_points],
            positions=[position.clone().detach() for position in self.positions],
            control_outputs=[control_output.clone().detach() for control_output in self.control_outputs],
            rbf_predictions=[rbf_pred.clone().detach() for rbf_pred in self.rbf_predictions],
            error_history=[error.clone().detach() for error in self.error_history],
            error_diff_history=[error_diff.clone().detach() for error_diff in self.error_diff_history],
            kp_values=[kp.clone().detach() for kp in self.kp_values],
            ki_values=[ki.clone().detach() for ki in self.ki_values],
            kd_values=[kd.clone().detach() for kd in self.kd_values],
            angle_history=[angle.clone().detach() for angle in self.angle_history],
            losses=[loss.clone().detach() for loss in self.losses],
            setpoints=[sp.clone().detach() for sp in self.setpoints]
        )

    def to_numpy(self) -> 'SimulationResults[np.ndarray]':
        results = self.detach_tensors()
        return SimulationResults(
            time_points=[time.numpy() for time in results.time_points],
            positions=[position.numpy() for position in results.positions],
            control_outputs=[control_output.numpy() for control_output in results.control_outputs],
            rbf_predictions=[rbf_pred.numpy() for rbf_pred in results.rbf_predictions],
            error_history=[error.numpy() for error in results.error_history],
            error_diff_history=[error_diff.numpy() for error_diff in results.error_diff_history],
            kp_values=[kp.numpy() for kp in results.kp_values],
            ki_values=[ki.numpy() for ki in results.ki_values],
            kd_values=[kd.numpy() for kd in results.kd_values],
            angle_history=[angle.numpy() for angle in results.angle_history],
            losses=[loss.numpy() for loss in results.losses],
            setpoints=[sp.numpy() for sp in results.setpoints]
        )
