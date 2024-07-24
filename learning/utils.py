import numpy as np

def calculate_angle_2p(pos1, pos2) -> np.float64:
    return np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0]) * 180 / np.pi