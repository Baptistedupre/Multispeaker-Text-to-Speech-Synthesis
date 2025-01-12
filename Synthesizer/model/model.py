import torch.nn as nn

from .modules import Encoder, Decoder


class TransformerTTS(nn.Module):
    def __init__(self):
        super(TransformerTTS, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, src_seq, src_pos, mel_input, mel_pos, speaker_embedding):
        enc_output, enc_mask, attention_enc = self.encoder(src_seq, speaker_embedding, src_pos) # noqa E501
        mel_output, out, gate_output, attention_dec, attention_enc_dec = self.decoder(enc_output, mel_pos, mel_input) # noqa E501

        output = mel_output, out, gate_output, attention_dec, attention_enc_dec, attention_enc # noqa E501 

        return output
