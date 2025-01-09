import torch
import torch.nn as nn
from math import sqrt
from torch.autograd import Variable
from torch.nn import functional as F

from params import hparams as hp
from layers import LinearNorm, ConvNorm, LocationLayer
from utils import to_gpu, get_mask_from_lengths


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
    def __init__(self):
        super(Attention, self).__init__()

        self.query_layer = LinearNorm(hp.model.attention_lstm_dim,
                                      hp.model.attention_dim,
                                      w_init_gain='tanh')
        self.memory_layer = LinearNorm(hp.model.encoder_embedding_dim,
                                       hp.model.attention_dim,
                                       w_init_gain='tanh')
        self.location_layer = LocationLayer(hp.model.attention_location_n_filters, # noqa E501
                                            hp.model.attention_location_kernel_size, # noqa E501
                                            hp.model.attention_dim)
        self.w = LinearNorm(hp.model.attention_dim, 1, w_init_gain='tanh')

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


class Prenet(nn.Module):
    def __init__(self):
        super(Prenet, self).__init__()

        self.fc1 = nn.Linear(hp.model.num_mels, hp.model.prenet_dim)
        self.fc2 = nn.Linear(hp.model.prenet_dim, hp.hp.model.prenet_dim)

    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=True)
        return F.dropout(F.relu(self.fc2(x)), p=0.5, training=True)


class Postnet(nn.Module):
    def __init__(self):
        super(Postnet, self).__init__()

        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hp.model.num_mels,
                         hp.model.postnet_embedding_dim,
                         kernel_size=hp.model.postnet_kernel_size, stride=1,
                         padding=int((hp.model.postnet_kernel_size - 1) / 2),
                         dilation=1),
                nn.BatchNorm1d(hp.model.postnet_embedding_dim)
            )
        )

        for i in range(1, hp.model.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hp.model.postnet_embedding_dim,
                             hp.model.postnet_embedding_dim,
                             kernel_size=hp.model.postnet_kernel_size, stride=1, # noqa E501
                             padding=int((hp.model.postnet_kernel_size - 1) / 2), # noqa E501
                             dilation=1),
                    nn.BatchNorm1d(hp.model.postnet_embedding_dim)
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hp.model.postnet_embedding_dim,
                         hp.model.num_mels,
                         kernel_size=hp.model.postnet_kernel_size, stride=1,
                         padding=int((hp.model.postnet_kernel_size - 1) / 2),
                         dilation=1),
                nn.BatchNorm1d(hp.model.num_mels)
            )
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), hp.model.p_decoder_dropout) # noqa E501

        return F.dropout(self.convolutions[-1](x), hp.model.p_decoder_dropout)


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()

        self.prenet = Prenet()

        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim,
            hparams.attention_rnn_dim)

        self.attention_layer = Attention()

        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + hparams.encoder_embedding_dim,
            hparams.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments
