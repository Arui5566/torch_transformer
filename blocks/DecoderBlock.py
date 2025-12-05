import torch
from torch import nn

from layers.SelfAttention import MultiHeadAttention
from layers.FeedForward import FFN


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, drop_prob=0.1):
        super().__init__()
        self.masked_mha = MultiHeadAttention(d_model, n_head)
        self.enc_dec_mha = MultiHeadAttention(d_model, n_head)
        self.ffn = FFN(d_model, d_ff, drop_prob)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        _x = x
        x = self.masked_mha(x, x, x, mask=tgt_mask)
        x = self.dropout(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.enc_dec_mha(x, enc_out, enc_out, mask=memory_mask)
        x = self.dropout(x)
        x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout(x)
        x = self.norm3(x + _x)

        return x
