import torch
import torch.nn as nn

from params import hparams as hp
from layers import LinearNorm, ConvNorm
    

class Postnet(nn.Module):
    def __init__(self):
        super(Postnet, self).__init__()

        convolutions = []
        for i in range(hp.model.postnet_n_convolutions - 1):
            conv_layer = nn.Sequential(
                ConvNorm(hp.model.symbols_embedding if i == 0 else hp.model.postnet_embedding,
                         hp.model.postnet_embedding,
                         kernel_size=hp.model.postnet_kernel_size, stride=1,
                         padding=int((hp.model.postnet_kernel_size - 1) / 2),
                         dilation=1),
                nn.Dropout(hp.model.p_postnet_dropout)
            )
            convolutions.append(conv_layer)
        convolutions.append(
            nn.Sequential(
                ConvNorm(hp.model.postnet_embedding, hp.model.n_mel_channels,
                         kernel_size=hp.model.postnet_kernel_size, stride=1,
                         padding=int((hp.model.postnet_kernel_size - 1) / 2),
                         dilation=1),
                nn.Dropout(hp.model.p_postnet_dropout)
            )
        )
        self.convolutions = nn.ModuleList(convolutions)

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = self.convolutions[i](x)
        return self.convolutions[-1](x)

class TacotronDecoder(nn.Module): 





class Tacotron(nn.Module): 
    