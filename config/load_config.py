import os

import yaml

from .config import cnfg
from .models import ConfigPack


def load_config(system: str) -> ConfigPack:
    yml_path = os.path.join(cnfg.YMLS_DIR, f"{system}.yml")
    with open(yml_path) as file:
        config_dict = yaml.safe_load(file)
    return ConfigPack.model_validate(config_dict)


def available_studies() -> list[str]:
    """Names of the study configurations in config/ymls, for CLI choices.

    Discovered rather than hardcoded so adding a study is one file, and so a
    typo in a name fails at argument parsing instead of at file open.
    """
    return sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(cnfg.YMLS_DIR)
        if name.endswith((".yml", ".yaml"))
    )
