import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    WEGHTS_DIR: str = 'weights'
    os.makedirs(WEGHTS_DIR, exist_ok=True)

    PUBLIC_DIR: str = 'public'
    os.makedirs(PUBLIC_DIR, exist_ok=True)

cnfg = Config()
