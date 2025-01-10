import torch
import numpy as np
from math import sqrt
import torch.nn as nn
from torch.nn import functional as F

from params import hparams as hp
from layers import Linear, Conv


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, n_position=200):
        super(PositionalEncoding, self).__init__()

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_model)) # noqa E501

    def _get_sinusoid_encoding_table(self, n_position, d_model):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_model) for hid_j in range(d_model)] # noqa E501

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) # noqa E501
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class EncoderPreNet(nn.Module):
    def __init__(self):
        super(EncoderPreNet, self).__init__()

        self.embedding = nn.Embedding(hp.model.n_symbols,
                                      hp.model.encoder_embedding_dim,
                                      padding_idx=0)

        convolutions = []
        for i in range(hp.model.encoder_n_convolutions):
            convolutions.append(
                Conv(hp.model.encoder_embedding_dim,
                     hp.model.encoder_embedding_dim,
                     kernel_size=hp.model.encoder_kernel_size,
                     padding=int(np.floor(hp.model.encoder_kernel_size / 2)),
                     w_init_gain='relu'))
        self.convolutions = nn.ModuleList(convolutions)

        self.batch_norm = nn.BatchNorm1d(hp.model.encoder_embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(hp.model.encoder_dropout)

        self.projection = Linear(hp.model.encoder_embedding_dim,
                                 hp.model.encoder_embedding_dim)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)

        for conv in self.convolutions:
            x = self.dropout(self.relu(self.batch_norm(conv(x))))

        x = x.transpose(1, 2)

        return self.projection(x)


class DecoderPreNet(nn.Module):
    def __init__(self):
        super(DecoderPreNet, self).__init__()

        self.fc1 = nn.Linear(hp.model.num_mels, hp.model.prenet_dim)
        self.fc2 = nn.Linear(hp.model.prenet_dim, hp.hp.model.prenet_dim)

        self.dropout = nn.Dropout(hp.model.p_decoder_dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        return self.dropout(self.relu(self.fc2(x)))


class Attention(nn.Module):
    def __init__(self, dim_key, attention_dropout=0.1):
        super(Attention, self).__init__()

        self.dim_key = dim_key
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, query, key, value, mask=None):
        attention = torch.matmul(query / sqrt(self.dim_key), key.transpose(2, 3)) # noqa E501

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)

        attention = self.dropout(F.softmax(attention, dim=-1))
        output = torch.matmul(attention, value)

        return output, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.dim_key = d_k
        self.dim_value = d_v

        self.query_layer = Linear(d_model, n_head * d_k)
        self.key_layer = Linear(d_model, n_head * d_k)
        self.value_layer = Linear(d_model, n_head * d_v)
        self.projection = Linear(n_head * d_v, d_model)

        self.attention = Attention(d_k)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, memory, decoder_output, mask=None):

        batch_size = memory.size(0)

        query = self.query_layer(x).view(batch_size, -1, self.n_head, self.dim_key) # noqa E501
        key = self.key_layer(x).view(batch_size, -1, self.n_head, self.dim_key) # noqa E501
        value = self.value_layer(decoder_output).view(batch_size, -1, self.n_head, self.dim_value) # noqa E501

        residual = query

        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2) # noqa E501

        if mask is not None:
            mask = mask.unsqueeze(1)

        x, attention = self.attention(query, key, value, mask=mask)
        x = x.transpose(1, 2).contiguous()

        x = x.view(batch_size, -1, self.n_head * self.dim_value)
        x = self.dropout(self.projectino(x))

        x += residual

        return self.layer_norm(x), attention


class FeedForwardNetwork(nn.Module):
    def __init__(self, dropout=0.5):
        super(FeedForwardNetwork, self).__init__()

        self.fc1 = Linear(hp.model.dim_model,
                          hp.model.dim_feedforward)

        self.fc2 = Linear(hp.model.dim_feedforward,
                          hp.model.dim_model)

        self.dropout = nn.Dropout(hp.model.dropout)
        self.layer_norm = nn.LayerNorm(hp.model.dim_model)

    def forward(self, x):
        residual = x
        x = self.fc2(self.relu(self.fc1(x)))
        x = self.dropout(x)

        x += residual
        x = self.layer_norm(x)

        return x


class Encoder(nn.Module):
    def __init__(self): 
        super(Encoder, self).__init__()

        self.encoder_prenet = EncoderPreNet()
        self.positional_encoding = PositionalEncoding(hp.model.dim_model)