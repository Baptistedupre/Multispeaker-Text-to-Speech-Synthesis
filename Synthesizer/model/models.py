import torch
import torch.nn as nn

from modules import Encoder, Decoder, Postnet


class Tacotron2(nn.Module):
    def __init__(self):
        super(Tacotron2, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = Postnet()

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs

        encoder_outputs = self.encoder(text_inputs, text_lengths)
        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mels, memory_lengths=text_lengths) # noqa E501
        mel_outputs_postnet = mel_outputs + self.postnet(mel_outputs)

        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

    def inference(self, inputs):
        encoder_outputs = self.encoder.inference(inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs) # noqa E501

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
