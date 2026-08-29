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

    #: Keeps a warm start off the flat ends of the sigmoid, where the head
    #: would receive almost no gradient and never recover.
    MIN_INITIAL_FRACTION = 0.05

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 3,
        num_layers: int = 1,
        dropout: float = 0.0,
        initial_gain_fraction: tuple[float, float, float] | None = None,
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

        # Small output weights so the network starts close to its bias, i.e. at
        # a definite, chosen controller rather than a random one.
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        self.set_initial_gains(initial_gain_fraction)

    def set_initial_gains(
        self, fraction: tuple[float, float, float] | None
    ) -> None:
        """Bias the head so the untrained network emits these gain fractions.

        Warm-starting at the classical tuning means training begins from a
        controller that already works and can only be judged by whether it
        improves on it. Starting from an arbitrary point instead makes the first
        several episodes a search for basic competence, and leaves the final
        comparison unable to separate "the scheduler learned something" from
        "the scheduler found its way back to roughly the classical gains".

        The requested fraction is clamped away from the ends of the range, not
        just away from 0 and 1. A sigmoid at 0.001 has a derivative of about
        1e-3, so a head warm-started there is effectively frozen: asking for
        Kd = 0 on the first-order thermal plant drove that output to a gradient
        norm of 2e-06 and it never moved again.

        Args:
            fraction: Desired ``gain / ceiling`` for each of Kp, Ki, Kd, in
                (0, 1). None leaves the head at the middle of its range.
        """
        if fraction is None:
            nn.init.zeros_(self.linear.bias)
            return
        target = torch.tensor(fraction, dtype=torch.float32).clamp(
            self.MIN_INITIAL_FRACTION, 1.0 - self.MIN_INITIAL_FRACTION
        )
        with torch.no_grad():
            self.linear.bias.copy_(torch.log(target / (1 - target)))  # logit

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
