import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        loss = torch.sqrt(self.mse(x, y))
        return loss
