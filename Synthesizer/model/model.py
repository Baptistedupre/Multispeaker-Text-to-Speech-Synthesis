import torch
import torch.nn as nn

from .modules import Encoder, Decoder


class TransformerTTS(nn.Module):
    def __init__(self):
        super(TransformerTTS, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, src_seq, src_pos, mel_input, mel_pos, speaker_embedding):
        self.initialize_masks(x_lengths=src_pos, y_lengths=mel_pos)
        enc_output, attention_enc = self.encoder(src_seq, speaker_embedding, mask=None, key_padding_mask=self.src_key_padding_mask) # noqa E501
        mel_output, out, gate_output, attention_dec, attention_enc_dec = self.decoder(enc_output, mel_input, src_mask=None, src_key_padding_mask=self.src_key_padding_mask, tgt_mask=self.tgt_mask, tgt_key_padding_mask=self.tgt_key_padding_mask) # noqa E501

        output = mel_output, out, gate_output, attention_dec, attention_enc_dec, attention_enc # noqa E501 

        return output

    def generate_square_subsequent_mask(self, lsz, rsz):
        return torch.triu(torch.ones(lsz, rsz) * float('-inf'), diagonal=1)

    def generate_padding_mask(self, lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(
            dtype=lengths.dtype, device=lengths.device)
        return ids >= lengths.unsqueeze(1).expand(-1, max_len)

    def initialize_masks(self, x_lengths=None, y_lengths=None):
        self.src_mask = None
        self.tgt_mask = None
        self.memory_mask = None
        self.src_key_padding_mask = None
        self.tgt_key_padding_mask = None
        if x_lengths is not None:
            S = x_lengths.max().item()
            self.src_mask = self.generate_square_subsequent_mask(S, S).to(device=x_lengths.device)         # noqa E501
            self.src_key_padding_mask = self.generate_padding_mask(x_lengths).to(device=x_lengths.device)  # noqa E501
        if y_lengths is not None:
            T = y_lengths.max().item()
            self.tgt_mask = self.generate_square_subsequent_mask(T, T).to(device=y_lengths.device)         # noqa E501
            self.tgt_key_padding_mask = self.generate_padding_mask(y_lengths).to(device=y_lengths.device)  # noqa E501
        if x_lengths is not None and y_lengths is not None:
            T = y_lengths.max().item()
            S = x_lengths.max().item()
            self.memory_mask = self.generate_square_subsequent_mask(T, S).to(device=y_lengths.device)      # noqa E501
