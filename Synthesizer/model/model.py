import torch
import torch.nn as nn

from .modules import Encoder, Decoder


class TransformerTTS(nn.Module):
    def __init__(self):
        super(TransformerTTS, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, text, pos_text, mel_input, pos_mel, speaker_embedding):

        enc_output, attention_enc, c_mask = self.encoder(text, speaker_embedding, pos_text) # noqa E501
        mel_output, out, gate_output, attention_dec, attention_enc_dec = self.decoder(enc_output, mel_input, pos_mel, c_mask) # noqa E501

        output = mel_output, out, gate_output, attention_dec, attention_enc_dec, attention_enc # noqa E501 

        return output
