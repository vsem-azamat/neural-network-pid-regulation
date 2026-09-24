"""LSTM that maps recent closed-loop history to PID gains."""

import math

import torch
from torch import Tensor, nn


class LSTMAdaptivePID(nn.Module):
    """Predicts PID gains from a window of loop history.

    **Residual mode (used by this project).** The network emits a bounded
    multiplicative correction around baseline gains:

        K = min(baseline * range^(2*sigmoid(z) - 1), ceiling)

    The head is zero-initialised, so the untrained scheduler *is* the baseline
    (the best constant gains it is later compared against) and everything it
    learns is a deviation from that point. Gains stay inside
    ``[baseline/range, baseline*range]`` whatever the weights, and the
    exponential form makes halving and doubling equal steps.

    **Absolute mode.** Without a baseline the sigmoid output is a fraction of
    the per-gain ceiling. The untrained network is then an arbitrary
    controller in the gain box; on the nonlinear trolley this mode ended up
    worse than a single constant.

    The baseline, ceiling and range are buffers, so a checkpoint carries its
    own output mapping. The sigmoid (not an unbounded ``exp``) keeps a drifting
    linear layer from producing huge gains and destabilising the loop.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int = 3,
        num_layers: int = 1,
        dropout: float = 0.0,
        baseline_gains: tuple[float, float, float] | None = None,
        gain_ceiling: tuple[float, float, float] | None = None,
        residual_range: float = 2.5,
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

        # Zero-initialise the head: the untrained network emits exactly its
        # centre (the baseline in residual mode, mid-box otherwise) instead of
        # a random offset from it. Gradients still reach the weights — the
        # sigmoid's derivative at 0.5 is its maximum.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Zero baseline = absolute mode. Buffers, not attributes, so the output
        # mapping travels inside the checkpoint.
        self.register_buffer("baseline", torch.zeros(output_size))
        self.register_buffer("ceiling", torch.ones(output_size))
        self.register_buffer("log_range", torch.tensor(0.0))
        if baseline_gains is not None:
            if gain_ceiling is None:
                raise ValueError("residual mode needs the gain ceiling as well")
            self.configure_residual(baseline_gains, gain_ceiling, residual_range)

    def configure_residual(
        self,
        baseline_gains: tuple[float, float, float],
        gain_ceiling: tuple[float, float, float],
        residual_range: float = 2.5,
    ) -> None:
        """Centre the head on ``baseline_gains``; see the class docstring."""
        if residual_range <= 1.0:
            raise ValueError(
                f"residual_range={residual_range} must exceed 1 — it is the "
                "multiplicative half-width of the correction band"
            )
        baseline = torch.tensor(baseline_gains, dtype=torch.float32)
        if not bool((baseline > 0).all()):
            raise ValueError(
                f"baseline gains must all be positive, got {baseline_gains} — "
                "a multiplicative correction around zero stays zero"
            )
        with torch.no_grad():
            self.baseline.copy_(baseline)
            self.ceiling.copy_(torch.tensor(gain_ceiling, dtype=torch.float32))
            self.log_range.fill_(math.log(residual_range))

    @property
    def residual_mode(self) -> bool:
        return bool((self.baseline > 0).any())

    @property
    def warm_start_gains(self) -> tuple[float, float, float] | None:
        """Gains the untrained head emits — what the PID should hold during
        warm-up, so handing control to the network is not a gain step."""
        if not self.residual_mode:
            return None
        return tuple(float(g) for g in self.baseline)

    def forward(
        self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Args:
            x: ``(batch, sequence_length, input_size)`` window of loop history.
            hidden: Previous LSTM state, or None to start fresh.

        Returns:
            ``(normalised_gains, hidden)``. Normalised gains are fractions of
            the per-gain ceiling; the caller multiplies by that same ceiling.
        """
        lstm_out, hidden = self.lstm(x, hidden)
        squashed = torch.sigmoid(self.linear(lstm_out[:, -1, :]))
        if not self.residual_mode:
            return squashed, hidden

        correction = torch.exp(self.log_range * (2.0 * squashed - 1.0))
        # The ceiling stays a hard promise: the fixed-gain searches this model
        # is compared against are confined to the same box, and letting the
        # network out of it would win by rule-breaking, not by scheduling.
        gains = torch.minimum(self.baseline * correction, self.ceiling)
        return gains / self.ceiling, hidden
