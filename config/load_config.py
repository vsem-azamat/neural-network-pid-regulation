import os
import yaml

from .config import cnfg
from .models import ConfigPack


def load_config(system: str) -> ConfigPack:
    yml_path = os.path.join(cnfg.YMLS_DIR, f"{system}.yml")
    with open(yml_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return ConfigPack.model_validate(config_dict)
