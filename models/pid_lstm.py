"""LSTM that maps recent closed-loop history to PID gains."""

import torch
from torch import Tensor, nn


class LSTMAdaptivePID(nn.Module):
    """Predicts normalised gains in (0, 1) from a window of loop history.

    The output is squashed with a sigmoid rather than an exponential. ``exp`` is
    positive but unbounded, so a modest drift in the linear layer's output
    produces enormous gains, the loop goes unstable, and the resulting gradient
    pushes the weights further out — the "numerical stability" failure mode this
    project ran into. A bounded head cannot diverge; the caller scales the
    (0, 1) output by a per-gain maximum, which also lets Kp, Ki and Kd have the
    very different magnitudes a real controller needs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 3,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.linear = nn.Linear(hidden_size, output_size)

        # Start near the middle of the allowed range: gains begin at half their
        # maximum instead of at an arbitrary corner of the output space.
        nn.init.zeros_(self.linear.bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)

    def forward(
        self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Args:
            x: ``(batch, sequence_length, input_size)`` window of loop history.
            hidden: Previous LSTM state, or None to start fresh.

        Returns:
            ``(normalised_gains, hidden)`` with gains in (0, 1).
        """
        lstm_out, hidden = self.lstm(x, hidden)
        return torch.sigmoid(self.linear(lstm_out[:, -1, :])), hidden
