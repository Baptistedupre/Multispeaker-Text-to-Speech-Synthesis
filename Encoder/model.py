import torch.nn as nn

from params import hparams as hp


class SpeakerEncoder(nn.Module):
    def __init__(self):
        super(SpeakerEncoder, self).__init__()

        # Define the network
        self.lstm = nn.LSTM(input_size=hp.data.nmels,
                            hidden_size=hp.model.hidden,
                            num_layers=hp.model.num_layers,
                            batch_first=True)
        self.linear = nn.Linear(in_features=hp.model.hidden,
                                out_features=hp.model.proj)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.linear(x)
        x = self.relu(x)
        return x
