import torch
from torch import nn

class RBFLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(RBFLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.centres, -1, 1)
        nn.init.constant_(self.sigmas, 1)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1) * self.sigmas.unsqueeze(0)
        return torch.exp(-distances)

class SystemRBFModel(nn.Module):
    def __init__(self, hidden_features=50):  # Increased hidden features
        super(SystemRBFModel, self).__init__()
        self.rbf = RBFLayer(in_features=4, out_features=hidden_features)  # Updated in_features to 4
        self.linear = nn.Linear(hidden_features, 1)

    def forward(self, x):
        x = self.rbf(x)
        x = self.linear(x)
        return x
