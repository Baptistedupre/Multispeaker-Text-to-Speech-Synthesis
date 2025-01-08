import torch
import torch.nn as nn

from params import hparams as hp
from layers import LinearNorm, ConvNorm



class LocationLayer(nn.Module):


    

class Prenet(nn.Module):
    def __init__(self):
        super(Prenet, self).__init__()

        self.layers = nn.ModuleList([
            nn.Linear(hp.model.encoder_embedding, hp.model.prenet_dim),
            nn.ReLU(),
            nn.Dropout(hp.model.p_prenet_dropout),
            nn.Linear(hp.model.prenet_dim, hp.model.prenet_dim),
            nn.ReLU(),
            nn.Dropout(hp.model.p_prenet_dropout)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    

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
    