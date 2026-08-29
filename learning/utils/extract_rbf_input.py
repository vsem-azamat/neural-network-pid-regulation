"""Feature extractors that turn a plant's live state into an RBF model input.

These run *inside* the differentiable simulation loop, so every tensor here has
to stay attached to the autograd graph.  Building the row with ``torch.stack``
rather than ``torch.tensor([...])`` is the whole point: ``torch.tensor`` copies
raw numbers into a fresh leaf and silently severs the backward path from the
loss to the LSTM that produced the control signal.

The extractors take the control signal *currently* being applied, so the
prediction lines up with the measurement one step later:

    rbf_prediction[k] == estimate of y[k + 1] == plant output after step k
"""

import torch
from torch import Tensor

from entities.systems import Thermal, Trolley


def _scalar(value) -> Tensor:
    """Coerce a state value to a 0-dim float tensor without breaking the graph."""
    tensor = value if isinstance(value, Tensor) else torch.tensor(float(value))
    return tensor.reshape(-1)[0].to(torch.float32)


def _row(values: list) -> Tensor:
    """Stack scalar features into a single (1, n_features) batch row."""
    return torch.stack([_scalar(v) for v in values]).unsqueeze(0)


def thermal(system: Thermal, control: Tensor) -> Tensor:
    """[temperature, dT/dt, heat input] -> next temperature."""
    return _row([system.X, system.dXdT, control])


def trolley(system: Trolley, control: Tensor) -> Tensor:
    """[position, velocity, acceleration, force] -> next position."""
    return _row([system.X, system.dXdT, system.d2XdT2, control])


EXTRACTORS = {"thermal": thermal, "trolley": trolley}
