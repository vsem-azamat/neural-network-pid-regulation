"""Typed configuration schema.

Feature counts are deliberately absent: the LSTM's input width is fixed by
:data:`learning.utils.N_FEATURES` and the RBF's by the plant's feature
extractor. Having them in the config only created a way for the two to disagree
— which they did, silently, because both plants happened to be set to 5.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Range(BaseModel):
    """An inclusive [low, high] interval."""

    low: float
    high: float

    @model_validator(mode="after")
    def _ordered(self) -> "Range":
        if self.high < self.low:
            raise ValueError(f"Range low={self.low} must not exceed high={self.high}")
        return self

    def as_tuple(self) -> tuple[float, float]:
        return self.low, self.high


class OptimizerConfig(BaseModel):
    lr: float
    momentum: float = Field(0.9)
    name: Literal["adam", "sgd"] = "adam"


class SchedulerConfig(BaseModel):
    gamma: float = 0.95


class ControlConfig(BaseModel):
    """Everything about the controller that is independent of the learner."""

    output_min: float
    output_max: float
    initial_gains: tuple[float, float, float]
    gain_ceiling: tuple[float, float, float]
    error_scale: float = Field(
        ..., description="Characteristic output magnitude, used to normalise "
        "the LSTM's input features."
    )
    tuning_step_input: float = Field(
        ..., description="Amplitude of the open-loop step test used by the "
        "classical tuning baseline."
    )


class ScenarioConfig(BaseModel):
    """How a training/evaluation episode is generated.

    A single constant setpoint per episode — what the original code produced,
    since `[value] * n` repeats one object — gives an adaptive controller nothing
    to adapt to. These knobs create reference changes, load disturbances and
    plant-to-plant variation, which is where gain scheduling can actually pay off.
    """

    setpoint: Range
    segment_time: Range = Field(
        ..., description="Seconds each setpoint is held before the next change."
    )
    disturbance_scale: float = Field(
        0.0, description="Std-dev of the load disturbance, in control units."
    )
    disturbance_hold: float = Field(
        1.0, description="Seconds a disturbance level is held."
    )
    randomize_plant: dict[str, Range] = Field(
        default_factory=dict,
        description="Per-episode plant parameter ranges (domain randomisation).",
    )


class LSTMModelConfig(BaseModel):
    hidden_size: int
    num_layers: int = 1
    dropout: float = 0.0


class LSTMConfig(BaseModel):
    train_time: float
    num_epochs: int
    sequence_length: int
    tbptt_window: int
    warm_up_steps: int
    grad_clip: float | None = 1.0
    loss_target: Literal["plant", "surrogate"] = "plant"
    overshoot_weight: float = 0.5
    effort_weight: float = Field(
        0.0,
        description="Penalty on control-signal movement. A scheduler that wins "
        "on tracking purely by working the actuator harder is not a clean win, "
        "so the trade-off is made explicit rather than left implicit.",
    )
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig = SchedulerConfig()
    model: LSTMModelConfig


class RBFModelConfig(BaseModel):
    hidden_size: int


class RBFConfig(BaseModel):
    lr: float
    num_epochs: int
    batch_size: int
    num_trajectories: int = Field(
        ..., description="Closed-loop-like rollouts used to build the dataset."
    )
    trajectory_steps: int
    validation_split: float = 0.2
    model: RBFModelConfig


class LearningConfig(BaseModel):
    dt: float
    lstm: LSTMConfig
    rbf: RBFConfig


class ConfigPack(BaseModel):
    """One study: a plant, how it is driven, and how the controller is trained.

    ``plant`` names the *dynamics class*, separately from the name of the config
    file. That lets a linear and a nonlinear study of the same plant live side by
    side and be compared, which is the whole point of having both.
    """

    plant: Literal["trolley", "thermal"]
    learning: LearningConfig
    control: ControlConfig
    scenario: ScenarioConfig
    system: dict[str, float]
