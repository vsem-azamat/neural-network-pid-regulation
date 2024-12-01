import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    WEIGHTS_DIR: str = "weights"
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    PUBLIC_DIR: str = "public"
    os.makedirs(PUBLIC_DIR, exist_ok=True)

    PLOTS_DIR: str = "plots"
    LEARNING_PLOTS: str = os.path.join(PLOTS_DIR, "learning")
    os.makedirs(LEARNING_PLOTS, exist_ok=True)
    SYSTEMS_PLOTS: str = os.path.join(PLOTS_DIR, "systems")
    os.makedirs(SYSTEMS_PLOTS, exist_ok=True)


cnfg = Config()
