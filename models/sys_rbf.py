"""Radial basis function network used as a differentiable plant surrogate."""

import torch
from torch import Tensor, nn


class RBFLayer(nn.Module):
    """Gaussian RBF layer: φ_j(x) = exp(−β_j · ‖x − c_j‖²).

    The width parameter is stored as ``log_beta`` and exponentiated on use. When
    it was stored raw, nothing stopped training from driving it negative — and a
    negative coefficient turns the Gaussian inside out, so activations grow
    without bound instead of decaying and the network diverges. Parameterising
    the log keeps β strictly positive for every value the optimiser can reach.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.empty(out_features, in_features))
        self.log_beta = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Inputs are normalised upstream, so centres spread over the unit cube
        # cover the data, and β = 1 gives basis functions of comparable width.
        nn.init.uniform_(self.centres, -1.0, 1.0)
        nn.init.zeros_(self.log_beta)

    @property
    def beta(self) -> Tensor:
        return torch.exp(self.log_beta)

    def forward(self, x: Tensor) -> Tensor:
        # (batch, 1, features) - (1, centres, features) -> (batch, centres)
        distances = (x.unsqueeze(1) - self.centres.unsqueeze(0)).pow(2).sum(-1)
        return torch.exp(-distances * self.beta.unsqueeze(0))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class SystemRBFModel(nn.Module):
    """One-step-ahead plant model: (state, control) -> next output.

    Normalisation statistics are registered as *buffers*, not parameters. As
    parameters they were trained along with the weights, so the mapping between
    raw and normalised units drifted during fitting while the saved dataset
    statistics stayed put — and they were also stored twice in the state dict.
    """

    def __init__(
        self,
        input_mean: Tensor,
        input_std: Tensor,
        output_mean: Tensor,
        output_std: Tensor,
        hidden_features: int = 50,
        input_size: int = 4,
        output_size: int = 1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.register_buffer("input_mean", torch.as_tensor(input_mean).float())
        self.register_buffer("input_std", torch.as_tensor(input_std).float() + eps)
        self.register_buffer("output_mean", torch.as_tensor(output_mean).float())
        self.register_buffer("output_std", torch.as_tensor(output_std).float() + eps)

        self.rbf = RBFLayer(in_features=input_size, out_features=hidden_features)
        self.linear = nn.Linear(hidden_features, output_size)

    @property
    def input_size(self) -> int:
        return self.rbf.in_features

    @property
    def hidden_features(self) -> int:
        return self.rbf.out_features

    @property
    def output_size(self) -> int:
        return self.linear.out_features

    def forward(self, x: Tensor) -> Tensor:
        x = (x - self.input_mean) / self.input_std
        x = self.rbf(x)
        x = self.linear(x)
        return x * self.output_std + self.output_mean
