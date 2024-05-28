import torch
from torch import Tensor
from dataclasses import dataclass

@dataclass
class Feedback:
    loss: Tensor
    
    Kp: Tensor
    Ki: Tensor
    Kd: Tensor
    
    Y: Tensor
    U: Tensor
    E: Tensor

    @classmethod
    def zeros(cls, length: int) -> 'Feedback':
        return Feedback(
            loss=torch.zeros(length),
            Kp=torch.zeros(length),
            Ki=torch.zeros(length),
            Kd=torch.zeros(length),
            Y=torch.zeros(length),
            U=torch.zeros(length),
            E=torch.zeros(length)
        )
