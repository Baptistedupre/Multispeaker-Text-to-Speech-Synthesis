import torch
import numpy as np
from math import sqrt, log
import torch.nn as nn
from torch.nn import functional as F

from text.symbols import symbols
from synthesizer_params import hparams as hp
from .layers import Linear, Conv


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 4096): # noqa E501
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-log(10000.0) / d_model)) # noqa E501
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x.transpose(0, 1))


class EncoderPreNet(nn.Module):
    def __init__(self):
        super(EncoderPreNet, self).__init__()

        self.embedding = nn.Embedding(len(symbols),
                                      hp.model.encoder_embedding_dim,
                                      padding_idx=0)
        std = sqrt(2.0 / (len(symbols) + hp.model.dim_model))
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)

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
        self.fc2 = nn.Linear(hp.model.prenet_dim, hp.model.prenet_dim)

        self.dropout = nn.Dropout(hp.model.p_decoder_dropout)
        self.relu = nn.ReLU()

        self.linear = nn.Linear(hp.model.prenet_dim, hp.model.dim_model)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.linear(x)


class Attention(nn.Module):
    def __init__(self, dim_key, attention_dropout=0.1):
        super(Attention, self).__init__()

        self.dim_key = dim_key
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, query, key, value, mask=None, key_padding_mask=None): # noqa E501
        attention = torch.matmul(query / sqrt(self.dim_key), key.transpose(2, 3)) # noqa E501
        if mask is not None:
            mask_expanded = mask[None, None, :, :]
            attention = attention.masked_fill(mask_expanded == 0, -1e9)
        elif key_padding_mask is not None:
            key_padding_mask_expanded = key_padding_mask[:, None, None, :]
            attention = attention.masked_fill(key_padding_mask_expanded[None, None, :, :] == 0, -1e9) # noqa E501

        attention = self.dropout(F.softmax(attention, dim=-1))
        output = torch.matmul(attention, value)

        return output, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.dim_key = d_k
        self.dim_value = d_v
        self.dim_model = d_model

        self.query_layer = Linear(d_k, n_head * d_model)
        self.key_layer = Linear(d_k, n_head * d_model)
        self.value_layer = Linear(d_v, n_head * d_model)
        self.projection = Linear(n_head * d_model, d_model)

        self.attention = Attention(d_k)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, memory, decoder_output, mask=None, key_padding_mask=None): # noqa E501

        batch_size = memory.size(0)
        residual = decoder_output

        query = self.query_layer(decoder_output).view(batch_size, -1, self.n_head, self.dim_model) # noqa E501
        key = self.key_layer(memory).view(batch_size, -1, self.n_head, self.dim_model) # noqa E501
        value = self.value_layer(memory).view(batch_size, -1, self.n_head, self.dim_model) # noqa E501

        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2) # noqa E501

        x, attention = self.attention(query, key, value, mask=mask, key_padding_mask=key_padding_mask) # noqa E501
        x = x.transpose(2, 3).contiguous()

        x = x.view(batch_size, -1, self.n_head * self.dim_model)
        x = self.dropout(self.projection(x))

        x += residual

        return self.layer_norm(x), attention


class FeedForwardNetwork(nn.Module):
    def __init__(self, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()

        self.fc1 = Linear(hp.model.dim_model,
                          hp.model.dim_feedforward)

        self.fc2 = Linear(hp.model.dim_feedforward,
                          hp.model.dim_model)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hp.model.dim_model)

    def forward(self, x):
        residual = x
        x = self.fc2(self.relu(self.fc1(x)))
        x = self.dropout(x)

        x += residual
        x = self.layer_norm(x)

        return x


class PostNet(nn.Module):
    def __init__(self, dropout=0.5):
        super(PostNet, self).__init__()

        convolutions = []
        convolutions.append(Conv(hp.model.num_mels,
                                 hp.model.dim_model,
                                 kernel_size=5,
                                 padding=int((5 - 1) / 2),
                                 stride=1,
                                 w_init_gain='tanh'))

        for i in range(1, hp.model.n_postnet_convolutions - 1):
            convolutions.append(Conv(hp.model.dim_model,
                                     hp.model.dim_model,
                                     kernel_size=5,
                                     padding=int((5 - 1) / 2),
                                     stride=1,
                                     w_init_gain='tanh'))

        convolutions.append(Conv(hp.model.dim_model,
                                 hp.model.num_mels,
                                 kernel_size=5,
                                 padding=int((5 - 1) / 2),
                                 stride=1,
                                 w_init_gain='linear'))
        self.convolutions = nn.ModuleList(convolutions)

        self.batch_norm = nn.BatchNorm1d(hp.model.dim_model)
        self.batch_norm_last = nn.BatchNorm1d(hp.model.num_mels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        for conv in self.convolutions[:-1]:
            x = self.dropout(torch.tanh(self.batch_norm(conv(x))))

        x = self.dropout(self.batch_norm_last(self.convolutions[-1](x)))
        x = x.transpose(1, 2)

        return x


class Encoder(nn.Module):
    def __init__(self, dropout=0.1):
        super(Encoder, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1))
        self.encoder_prenet = EncoderPreNet()
        self.linear_projection = Linear(hp.model.encoder_embedding_dim + hp.model.speaker_embedding_dim, hp.model.dim_model) # noqa E501
        self.positional_encoding = PositionalEncoding(hp.model.dim_model)
        self.multihead_attention = MultiHeadAttention(hp.model.n_head,
                                                      hp.model.dim_model,
                                                      hp.model.dim_model, # noqa E501
                                                      hp.model.dim_model, # noqa E501
                                                      dropout) # noqa E501
        self.feed_forward = FeedForwardNetwork()
        self.pos_dropout = nn.Dropout(dropout)

    def forward(self, x, speaker_embedding, mask=None, key_padding_mask=None): # noqa E501
        x = self.encoder_prenet(x)

        x = self.positional_encoding(x)

        x = self.add_speaker_embedding(x, speaker_embedding)
        x = self.linear_projection(x)
        x, attention = self.multihead_attention(x, x, mask=mask, key_padding_mask=key_padding_mask) # noqa E501
        x = self.feed_forward(x)

        return x, attention

    def add_speaker_embedding(self, x, speaker_embedding):
        batch_size = x.size()[0]
        num_chars = x.size()[1]

        if speaker_embedding.dim() == 1:
            idx = 0
        else:
            idx = 1

        speaker_embedding_size = speaker_embedding.size()[idx]
        e = speaker_embedding.repeat_interleave(num_chars, dim=idx)

        e = e.reshape(batch_size, speaker_embedding_size, num_chars)
        e = e.transpose(1, 2)

        x = torch.cat((x, e), 2)
        return x


class Decoder(nn.Module):
    def __init__(self, dropout=0.1):
        super(Decoder, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1))
        self.decoder_prenet = DecoderPreNet()
        self.positional_encoding = PositionalEncoding(hp.model.dim_model)
        self.pos_dropout = nn.Dropout(dropout)
        self.masked_multihead_attention = MultiHeadAttention(hp.model.n_head,
                                                             hp.model.dim_model, # noqa E501
                                                             hp.model.dim_model, # noqa E501
                                                             hp.model.dim_model, # noqa E501
                                                             dropout)
        self.multihead_attention = MultiHeadAttention(hp.model.n_head,
                                                      hp.model.dim_model,
                                                      hp.model.dim_model,
                                                      hp.model.dim_model,
                                                      dropout)
        self.feed_forward = FeedForwardNetwork()
        self.mel_linear = Linear(hp.model.dim_model, hp.model.num_mels)
        self.gate_linear = Linear(hp.model.dim_model, 1)
        self.postnet = PostNet()

    def forward(self, encoder_output, decoder_input, src_mask=None, src_key_padding_mask=None, tgt_mask=None, tgt_key_padding_mask=None): # noqa E501

        decoder_input = self.decoder_prenet(decoder_input)

        decoder_input = self.positional_encoding(decoder_input)

        decoder_output, attention_dec = self.masked_multihead_attention(decoder_input, decoder_input, mask=tgt_mask, key_padding_mask=tgt_key_padding_mask) # noqa E501
        decoder_output, attention_enc = self.multihead_attention(encoder_output, decoder_output, key_padding_mask=src_key_padding_mask) # noqa E501
        decoder_output = self.feed_forward(decoder_output)

        mel_output = self.mel_linear(decoder_output)
        gate_output = self.gate_linear(decoder_output)
        mel_output_postnet = self.postnet(mel_output)

        out = mel_output + mel_output_postnet

        return mel_output, out, gate_output, attention_dec, attention_enc
