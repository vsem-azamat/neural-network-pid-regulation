"""Builds the history window the gain-scheduling LSTM sees.

One implementation serves both plants. The features are dimensionless:

    0  e / error_scale                normalised tracking error
    1  Δe / error_scale               normalised error rate
    2  Kp / Kp_max                    current gains, as a fraction of their
    3  Ki / Ki_max                    allowed range
    4  Kd / Kd_max
    5  (y − mid) / halfspan           operating point
    6  (r − mid) / halfspan           commanded operating point
    7  u / control_scale              last control signal

Normalisation lets one architecture serve a plant in metres and one in kelvin.
Features 5–6 are what gain scheduling keys on: on a nonlinear plant the right
gains depend on where the plant is, and without them the network sees the
same input at both ends of the range. Feature 7 reveals actuator saturation,
which time-varying gains can exploit even on a linear plant.
"""

import torch
from torch import Tensor

from classes.simulation import SimulationConfig, SimulationResults

N_FEATURES = 8


def extract_lstm_input(
    simulation_config: SimulationConfig, results: SimulationResults
) -> Tensor:
    """Return a ``(1, sequence_length, N_FEATURES)`` window, zero-padded at the start.

    ``sequence_length`` trades off against the recurrence, and running both at
    once is waste: with a 40-sample window the carried hidden state changes the
    predicted gains by 0.0 (max 6e-8 over 200 steps), because the window already
    contains everything the hidden state could remember. That is 40x the work per
    control step for no effect, and it means the LSTM is not really being used as
    one.

    Short windows put the memory back in the hidden state, where a recurrent
    network's memory belongs, and truncated BPTT still propagates gradients
    through it across the whole window.
    """
    length = simulation_config.sequence_length
    window = torch.zeros(length, N_FEATURES)

    available = min(length, len(results.error_history))
    if available == 0:
        return window.unsqueeze(0)

    error_scale = max(abs(simulation_config.error_scale), 1e-6)
    gain_scale = simulation_config.gain_scale.clamp(min=1e-6)

    def recent(name: str) -> Tensor:
        values = getattr(results, name)[-available:]
        return torch.stack([torch.as_tensor(v).reshape(-1)[0] for v in values]).detach()

    window[-available:, 0] = recent("error_history") / error_scale
    window[-available:, 1] = recent("error_diff_history") / error_scale
    window[-available:, 2] = recent("kp_values") / gain_scale[0]
    window[-available:, 3] = recent("ki_values") / gain_scale[1]
    window[-available:, 4] = recent("kd_values") / gain_scale[2]

    midpoint = simulation_config.operating_midpoint
    halfspan = simulation_config.operating_halfspan
    window[-available:, 5] = (recent("positions") - midpoint) / halfspan
    window[-available:, 6] = (recent("setpoints") - midpoint) / halfspan

    control_scale = max(abs(simulation_config.control_scale), 1e-6)
    window[-available:, 7] = recent("control_outputs") / control_scale

    # The window is an observation, not part of the differentiable control path:
    # gradients reach the LSTM through its *output* (the gains it sets), which is
    # what the tracking loss actually depends on.
    return window.unsqueeze(0)
