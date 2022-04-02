import torch
from torch import nn

class Charbonnier(nn.Module):
    def __init__(self, epsilon=0.001):
        super(Charbonnier, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, gt):
        return torch.mean(torch.sqrt((output - gt) ** 2 + self.epsilon ** 2))
