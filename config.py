import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    WEIGHTS_DIR: str = 'weights'
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    PUBLIC_DIR: str = 'public'
    os.makedirs(PUBLIC_DIR, exist_ok=True)

    PLOTS_DIR: str = 'plots'
    os.makedirs(PLOTS_DIR, exist_ok=True)

cnfg = Config()
