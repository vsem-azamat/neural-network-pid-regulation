import os
import torch
import pickle
from config import cnfg
from models.sys_rbf import SystemRBFModel


def load_model(
    model: torch.nn.Module, name: str, weights_only: bool = False
) -> torch.nn.Module:
    path = os.path.join(cnfg.WEIGHTS_DIR, name)
    model.load_state_dict(torch.load(path, weights_only=weights_only))
    return model


def save_model(model, name: str) -> None:
    path = os.path.join(cnfg.WEIGHTS_DIR, name)
    torch.save(model.state_dict(), path)


def load_pickle(name):
    path = os.path.join(cnfg.WEIGHTS_DIR, name)
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(data, name) -> None:
    path = os.path.join(cnfg.WEIGHTS_DIR, name)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def save_rbf_model(model: SystemRBFModel, name: str) -> None:
    path = os.path.join(cnfg.WEIGHTS_DIR, name)
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_mean": model.input_norm.mean,
        "input_std": model.input_norm.std,
        "output_mean": model.output_denorm.mean,
        "output_std": model.output_denorm.std,
        "hidden_features": model.rbf.out_features,
        "input_size": model.rbf.in_features,
        "output_size": model.linear.out_features,
    }
    torch.save(checkpoint, path)


def load_rbf_model(name: str) -> SystemRBFModel:
    path = os.path.join(cnfg.WEIGHTS_DIR, name)
    checkpoint = torch.load(path, weights_only=True)
    model = SystemRBFModel(
        input_size=checkpoint["input_size"],
        output_size=checkpoint["output_size"],
        hidden_features=checkpoint["hidden_features"],
        input_mean=checkpoint["input_mean"],
        input_std=checkpoint["input_std"],
        output_mean=checkpoint["output_mean"],
        output_std=checkpoint["output_std"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model
