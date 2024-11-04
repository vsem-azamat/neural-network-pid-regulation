import torch
from torch import nn


class LSTMAdaptivePID(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMAdaptivePID, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)
        pid_params = self.linear(lstm_out[:, -1, :])
        return torch.exp(pid_params), hidden  # Ensure positive PID parameters
