import torch
from torch import nn
import math

class TokenEmbedding(nn.Embedding):
    """
    x: [batch_size, seq_len] -> te: [batch_size, seq_len, d_model]
    """
    def __init__(self, voc_size, d_model):
        super(TokenEmbedding, self).__init__(voc_size, d_model)

class PositionEmbedding(nn.Module):
    """
    x: [batch_size, seq_len, d_model] -> pe: [batch_size, seq_len, d_model]
    """
    def __init__(self, max_len, d_model):
        super(PositionEmbedding, self).__init__()
        pe = torch.zeros((max_len, d_model))
        pe.requires_grad = False

        pos = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            /d_model
            *-math.log(10000.0)
        )

        pe[:, 0::2] = torch.sin(pos / div_term)
        pe[:, 1::2] = torch.cos(pos / div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        batch_size, seq_size, d_model = x.shape

        return self.pe[:, :seq_size]
