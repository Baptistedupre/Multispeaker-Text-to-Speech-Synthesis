import torch 
import torch.nn as nn

class GE2ELoss(nn.Module):
    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)

    def sim_matrix(self, embeddings):
        """
        Args:
            embeddings: shape (n_speakers, n_utterances, embedding_size)
        Returns:
            similarity_matrix: shape (n_speakers, n_utterances)
        """
        


    def loss(self, sim_matrix):
        loss = -torch.log(sim_matrix.diag())
        loss = loss.mean()
        return loss
    
    def forward(self, x):
