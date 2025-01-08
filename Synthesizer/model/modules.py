import torch
import torch.nn as nn
from torch.nn import functional as F

from params import hparams as hp
from layers import LinearNorm, ConvNorm


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding

        convolutions = []
        for i in range(hp.model.encoder_n_convolutions):
            convolutions.append(ConvNorm(hp.model.symbols_embedding,
                                hp.model.encoder_embedding,
                                kernel_size=hp.model.encoder_kernel_size,
                                stride=1, padding=int((hp.model.encoder_kernel_size - 1) / 2), # noqa E501
                                dilation=1))
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hp.model.encoder_embedding,
                            int(hp.model.encoder_embedding / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, speaker_embedding=None):
        x = self.embedding(x)
        for conv in self.convolutions:
            x = conv(x)
        x = x.transpose(1, 2)
        x = self.lstm(x)
        if speaker_embedding is not None:
            x = self.add_speaker_embedding(x, speaker_embedding)
        return x

    def add_speaker_embedding(self, x, speaker_embedding):

        batch_size = x.size(0)
        num_chars = x.size(1)

        if speaker_embedding.dim() == 1:
            idx = 0
        else:
            idx = 1

        speaker_embedding_size = speaker_embedding.size(idx)
        e = speaker_embedding.repeat_interleave(num_chars, dim=idx)

        e = e.reshape(batch_size, speaker_embedding_size, num_chars)
        e = e.transpose(1, 2)

        return torch.cat((x, e), dim=2)





class Attention(nn.Module):
    def __init__(self, attention_lstm_dim, attention_dim, 
                 encoder_embedding_dim, attention_location_n_filters,
                 attention_location_kernel_size):
        super(Attention, self).__init__()

        self.query_layer = LinearNorm(attention_lstm_dim,
                                      attention_dim,
                                      w_init_gain='tanh')
        self.memory_layer = LinearNorm(encoder_embedding_dim,
                                       attention_dim,
                                       w_init_gain='tanh')
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.w = LinearNorm(attention_dim, 1, w_init_gain='tanh')

    def alignement_energies(self, query, memory, attention_weights_cat):

        key = self.memory_layer(memory)
        query = self.query_layer(query)
        location = self.location_layer(attention_weights_cat)

        energies = self.w(torch.tanh(query + key + location))

        return energies.squeeeze(-1)

    def forward(self, query, memory, attention_weights_cat):

        alignement = self.alignement_energies(query, memory, attention_weights_cat) # noqa E501

        attention_weights = F.softmax(alignement, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1) # noqa E501

        return attention_context, attention_weights
