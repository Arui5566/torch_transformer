from layers.SelfAttention import MultiHeadAttention
from layers.FeedForward import FFN
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, drop_prob=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_head)
        self.ffn = FFN(d_model, d_ff, drop_prob)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        attn_out = self.mha(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x