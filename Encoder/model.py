import torch
import torch.nn as nn
import numpy as np

from encoder.params import * 

class SpeakerEncoder(nn.Module):
    def __init__(self, device, loss_device):
        super().__init__()
        self.loss_device = loss_device

        # Define the network
        self.lstm = nn.LSTM(input_size=mel_n_channels, 
                            hidden_size=hidden_size, 
                            num_layers=lstm_num_layers, 
                            batch_first=True)
        self.linear = nn.Linear(in_features=model_hidden_size, 
                                out_features=model_embedding_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.linear(x)
        x = self.relu(x)
        return x
        