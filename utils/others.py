import torch
import numpy as np

def tensor2list(tensor) -> list[float | int]:
    try:
        return [t.item() for t in tensor]
    except AttributeError:
        return [t for t in tensor]


def calculate_angle_2p(pos1, pos2):
    return torch.atan((pos2[1] - pos1[1]) / (pos2[0] - pos1[0])) * 180 / np.pi
