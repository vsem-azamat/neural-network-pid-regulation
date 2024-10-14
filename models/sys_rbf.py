import torch
from torch import nn

class InputNormalizationLayer(nn.Module):
    def __init__(self, mean, std):
        super(InputNormalizationLayer, self).__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std + 1e-8)  # Add epsilon to avoid division by zero

    def forward(self, x):
        return (x - self.mean) / self.std

class OutputDenormalizationLayer(nn.Module):
    def __init__(self, mean, std):
        super(OutputDenormalizationLayer, self).__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std + 1e-8)

    def forward(self, x):
        return x * self.std + self.mean

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
    def __init__(self, 
        input_mean,
        input_std, 
        output_mean, 
        output_std, 
        hidden_features=50, 
        input_size=4, 
        output_size=1) -> None:
        super(SystemRBFModel, self).__init__()
        self.input_mean = nn.Parameter(input_mean, requires_grad=False)
        self.input_std = nn.Parameter(input_std, requires_grad=False)
        self.output_mean = nn.Parameter(output_mean, requires_grad=False)
        self.output_std = nn.Parameter(output_std, requires_grad=False)
        
        self.input_norm = InputNormalizationLayer(self.input_mean, self.input_std)
        self.rbf = RBFLayer(in_features=input_size, out_features=hidden_features)
        self.linear = nn.Linear(hidden_features, output_size)
        self.output_denorm = OutputDenormalizationLayer(self.output_mean, self.output_std)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.rbf(x)
        x = self.linear(x)
        x = self.output_denorm(x)
        return x
