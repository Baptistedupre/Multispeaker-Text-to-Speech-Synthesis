import torch
import torch.nn as nn


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()

        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim): # noqa E501
        super(LocationLayer, self).__init__()

        self.location_conv = ConvNorm(1, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=int((attention_kernel_size - 1) / 2), # noqa E501
                                      stride=1, dilation=1)
        self.location_dense = LinearNorm(attention_n_filters,
                                         attention_dim,
                                         w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        attention = self.location_conv(attention_weights_cat)
        attention = attention.transpose(1, 2)
        attention = self.location_dense(attention)
        return attention
