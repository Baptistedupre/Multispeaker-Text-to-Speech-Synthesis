import torch
import torch.nn as nn

from params import hparams as hp


class SpeakerEncoder(nn.Module):
    def __init__(self):
        super(SpeakerEncoder, self).__init__()

        # Define the network
        self.lstm = nn.LSTM(input_size=hp.data.nmels,
                            hidden_size=hp.model.hidden,
                            num_layers=hp.model.nb_layers,
                            batch_first=True)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.linear = nn.Linear(in_features=hp.model.hidden,
                                out_features=hp.model.proj)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, x.size(1) - 1]
        x = self.linear(x)
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x
