"""Checkpoint helpers."""

import os

import torch

from config import cnfg
from models.sys_rbf import SystemRBFModel


def _path(name: str) -> str:
    return os.path.join(cnfg.WEIGHTS_DIR, name)


def load_model(model: torch.nn.Module, name: str) -> torch.nn.Module:
    model.load_state_dict(torch.load(_path(name), weights_only=True))
    model.eval()
    return model


def save_model(model: torch.nn.Module, name: str) -> None:
    torch.save(model.state_dict(), _path(name))


def save_rbf_model(model: SystemRBFModel, name: str) -> None:
    """Save weights plus the shape needed to rebuild the model.

    Normalisation statistics live in the state dict as buffers, so they no
    longer need to be stored separately.
    """
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_features": model.hidden_features,
            "input_size": model.input_size,
            "output_size": model.output_size,
        },
        _path(name),
    )


def load_rbf_model(name: str) -> SystemRBFModel:
    checkpoint = torch.load(_path(name), weights_only=True)
    state = checkpoint["state_dict"]
    model = SystemRBFModel(
        input_size=checkpoint["input_size"],
        output_size=checkpoint["output_size"],
        hidden_features=checkpoint["hidden_features"],
        input_mean=state["input_mean"],
        input_std=state["input_std"],
        output_mean=state["output_mean"],
        output_std=state["output_std"],
        eps=0.0,  # the saved std already includes it
    )
    model.load_state_dict(state)
    model.eval()
    return model
