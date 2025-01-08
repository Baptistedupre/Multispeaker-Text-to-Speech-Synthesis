import torch
import torch.nn as nn


class TacotronLoss(nn.Module):
        def __init__(self):
            super(TacotronLoss, self).__init__()
            self.mse_loss = nn.MSELoss()
            self.bce_loss = nn.BCELoss()
