from torch import nn


class TransformerTTSLoss(nn.Module):
    def __init__(self, r_gate=5.):
        super(TransformerTTSLoss, self).__init__()

        self.mel_loss = nn.MSELoss()
        self.gate_loss = nn.BCEWithLogitsLoss()
        self.r_gate = r_gate

    def forward(self, mel_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False

        mel_out_postnet, mel_out, gate_out = mel_output
        gate_out = gate_out.view(-1, 1)

        mel_loss = self.mel_loss(mel_out, mel_target) + self.mel_loss(mel_out_postnet, mel_target) # noqa E501
        gate_loss = self.gate_loss(gate_out, gate_target) * self.r_gate

        return mel_loss + gate_loss
