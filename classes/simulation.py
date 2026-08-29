"""Configuration and result containers for one simulation episode."""

from dataclasses import dataclass, field, fields
from typing import Generic, TypeVar

import numpy as np
import torch

T = TypeVar("T", torch.Tensor, np.ndarray)


@dataclass
class LearningConfig:
    """Top-level knobs for a training run."""

    dt: torch.Tensor
    num_epochs: int
    train_time: float
    learning_rate: float

    @property
    def train_steps(self) -> int:
        return int(self.train_time / float(self.dt))


@dataclass
class SimulationConfig(Generic[T]):
    """Configuration for a single simulation episode.

    Attributes:
        setpoints: Reference value at each step. One entry per step, so a
            varying reference is expressed simply by varying this list.
        dt: Time step.
        sequence_length: Number of past samples fed to the LSTM.
        tbptt_window: Steps between truncated-backprop updates during training.
        warm_up_steps: Steps to run on the initial gains before the LSTM takes
            over, so it sees some history before its first prediction.
        pid_gain_factor: Maximum value for each gain. The network emits
            normalised gains in (0, 1); this is what they are scaled by. Pass a
            single number for a shared ceiling, or ``(kp_max, ki_max, kd_max)`` —
            Kp, Ki and Kd usually differ by orders of magnitude, and one shared
            ceiling forces them into the same range.
        error_scale: Characteristic magnitude of the plant output, used to
            normalise the features fed to the LSTM. Without it the network sees
            hundreds of kelvin for one plant and single-digit metres for the
            other, and the same architecture cannot serve both.
        disturbances: Optional load disturbance per step, same units as the
            control signal.
    """

    setpoints: list[T]
    dt: T
    sequence_length: int = 50
    tbptt_window: int = 25
    warm_up_steps: int = 10
    pid_gain_factor: float | tuple[float, float, float] = 15.0
    error_scale: float = 1.0
    disturbances: list[T] | None = None

    @property
    def num_steps(self) -> int:
        return len(self.setpoints)

    @property
    def gain_scale(self) -> torch.Tensor:
        """Per-gain ceiling as a (3,) tensor."""
        factor = self.pid_gain_factor
        if isinstance(factor, (int, float)):
            factor = (float(factor),) * 3
        return torch.tensor(factor, dtype=torch.float32)

    def disturbance_at(self, step: int) -> torch.Tensor:
        if self.disturbances is None:
            return torch.tensor(0.0)
        return torch.as_tensor(self.disturbances[step], dtype=torch.float32)


@dataclass
class SimulationResults(Generic[T]):
    """Per-step history of one episode.

    ``rbf_predictions[k]`` is the surrogate's estimate of ``positions[k]``: both
    refer to the plant output *after* step ``k``, so the two can be compared
    directly without an off-by-one correction.
    """

    time_points: list[T] = field(default_factory=list)
    setpoints: list[T] = field(default_factory=list)
    positions: list[T] = field(default_factory=list)
    control_outputs: list[T] = field(default_factory=list)
    rbf_predictions: list[T] = field(default_factory=list)
    error_history: list[T] = field(default_factory=list)
    error_diff_history: list[T] = field(default_factory=list)
    kp_values: list[T] = field(default_factory=list)
    ki_values: list[T] = field(default_factory=list)
    kd_values: list[T] = field(default_factory=list)
    disturbances: list[T] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)

    _TENSOR_FIELDS = (
        "time_points",
        "setpoints",
        "positions",
        "control_outputs",
        "rbf_predictions",
        "error_history",
        "error_diff_history",
        "kp_values",
        "ki_values",
        "kd_values",
        "disturbances",
    )

    def __len__(self) -> int:
        return len(self.positions)

    def detach_all(self) -> "SimulationResults":
        """Detach every stored tensor in place.

        Called at each truncation boundary: once ``backward()`` has freed the
        graph for a window, any tensor still pointing into it would raise on the
        next backward pass.
        """
        for name in self._TENSOR_FIELDS:
            values = getattr(self, name)
            setattr(
                self,
                name,
                [v.detach() if isinstance(v, torch.Tensor) else v for v in values],
            )
        return self

    def to_numpy(self) -> "SimulationResults[np.ndarray]":
        """Return a detached NumPy copy, leaving this instance untouched."""
        out = SimulationResults()
        for f in fields(self):
            values = getattr(self, f.name)
            if f.name in self._TENSOR_FIELDS:
                values = [
                    v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                    for v in values
                ]
            else:
                values = list(values)
            setattr(out, f.name, values)
        return out

    def as_floats(self, name: str) -> list[float]:
        """One field as plain Python floats, for metrics and plotting."""
        return [
            float(v.detach()) if isinstance(v, torch.Tensor) else float(v)
            for v in getattr(self, name)
        ]
