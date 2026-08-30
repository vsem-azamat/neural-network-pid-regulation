"""Builds the history window the gain-scheduling LSTM sees.

One implementation serves both plants. Previously each training script carried
its own near-identical copy, and the comparison code imported the *trolley*
one and used it for the thermal plant too — which only worked by coincidence,
because both happened to be configured with five input features.

The features are dimensionless by construction:

    0  e / error_scale                normalised tracking error
    1  Δe / error_scale               normalised error rate
    2  Kp / Kp_max                    current gains, as a fraction of their
    3  Ki / Ki_max                    allowed range
    4  Kd / Kd_max
    5  (y − mid) / halfspan           operating point
    6  (r − mid) / halfspan           commanded operating point

Feeding raw signals instead — as the original did, with absolute positions and
setpoints — means the network sees inputs in the hundreds for the thermal plant
and single digits for the trolley, so one architecture cannot serve both and the
first layer spends its capacity undoing the scale.

Features 5 and 6 are the *operating point*, and without them a gain scheduler
cannot do the one thing gain scheduling is for. On a nonlinear plant the right
gains are a function of where the plant is sitting: the hardening spring is 14
times stiffer at the end of its travel than at the origin, and the radiative
thermal plant loses heat three times faster at 500 K than at 320 K. An
error-only feature set is invariant to all of that — the network sees the same
input at both ends of the range and must emit the same gains, so it is
structurally incapable of scheduling however long it trains. Normalising them
against the study's own operating range keeps them dimensionless, so one
architecture still serves a plant in metres and a plant in kelvin.
"""

import torch
from torch import Tensor

from classes.simulation import SimulationConfig, SimulationResults

N_FEATURES = 7


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

    # The window is an observation, not part of the differentiable control path:
    # gradients reach the LSTM through its *output* (the gains it sets), which is
    # what the tracking loss actually depends on.
    return window.unsqueeze(0)
