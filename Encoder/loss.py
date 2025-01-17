import torch
import torch.nn as nn

from utils import similarity_matrix, calc_loss


class GE2ELoss(nn.Module):
    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True) # noqa E501
        self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True) # noqa E501
        self.device = device

    def forward(self, embeddings):
        """
        loss_matrix = torch.zeros(embeddings.shape[:2])
        sim_matrix = similarity_matrix(embeddings, self.w, self.b, self.device)
        loss, _ = calc_loss(sim_matrix)
        """
        torch.clamp(self.w, 1e-6)
        sim_matrix = similarity_matrix(embeddings, self.w, self.b, self.device) # noqa E501
        loss, _ = calc_loss(sim_matrix)
        return loss
