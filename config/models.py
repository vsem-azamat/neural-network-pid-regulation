from pydantic import BaseModel, Field


class OptimizerConfig(BaseModel):
    lr: float
    momentum: float = Field(0.9)


class SchedulerConfig(BaseModel):
    gamma: float


class ModelConfig(BaseModel):
    input_size: int
    hidden_size: int
    output_size: int


class LSTMConfig(BaseModel):
    train_time: float
    num_epochs: int
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    model: ModelConfig
    sequence_length: int
    sequence_step: int
    pid_gain_factor: int


class RBFConfig(BaseModel):
    lr: float
    num_epochs: int
    batch_size: int
    model: ModelConfig


class LearningConfig(BaseModel):
    dt: float
    lstm: LSTMConfig
    rbf: RBFConfig


class ConfigPack(BaseModel):
    learning: LearningConfig
    system: dict[str, float]
