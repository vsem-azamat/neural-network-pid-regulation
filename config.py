import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    WEGHTS_DIR: str = 'weights'
    os.makedirs(WEGHTS_DIR, exist_ok=True)

cnfg = Config()
