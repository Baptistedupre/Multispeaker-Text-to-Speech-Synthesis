import torch
import torch.nn as nn


class Linear(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(Linear, self).__init__()

        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class Conv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(Conv, self).__init__()

        self.conv_layer = nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride, padding=padding)

        nn.init.xavier_uniform_(
            self.conv_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.conv_layer(x)
