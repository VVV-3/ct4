import torch.nn as nn
import numpy as np



class Lstm(nn.Module):
    """lstm+mlp"""
    def __init__(self, inp_dim, hidden_dim, target_size=2):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(inp_dim, hidden_dim)
        self.mlp  = nn.Linear(hidden_dim, target_size)
        self.sigm = nn.Softmax(-1)
    def forward(self, x):
        _, hidden = self.lstm( x )
        x = self.mlp(hidden[1])
        x = self.sigm(x)
        return x