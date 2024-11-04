import torch.nn as nn
from torch import Tensor


class PIDNN(nn.Module):
    def __init__(self):
        super(PIDNN, self).__init__()
        self.fc0 = nn.LayerNorm(2)
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 3)

        # Normalization
        self.norm = nn.functional.normalize

        # Activation functions
        self.act_tahn = nn.Tanh()
        self.act_relu = nn.ReLU()

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        # Initialize biases
        self.fc1.bias.data.fill_(0.1)
        self.fc2.bias.data.fill_(0.1)

    def forward(self, x) -> Tensor:
        # x = self.norm(x, dim=0, p=2)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.act_relu(x)
        x = self.fc2(x)
        x = self.act_tahn(x)
        return x
