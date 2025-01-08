import torch
import torch.nn as nn

from params import hparams as hp
from layers import LinearNorm, ConvNorm


class Tacotron2(nn.Module): 
