from layers.Embed import TokenEmbedding, PositionEmbedding
from blocks.EncoderBlock import TransformerBlock
from torch import nn

class TransformerEncoder(nn.Module):
    def __init__(self, voc_size, d_model, d_ff,  n_head, n_layers, max_len= 512, drop_prob= 0.1,):
        super().__init__()
        self.token_emb = TokenEmbedding(voc_size, d_model)
        self.pos_emb = PositionEmbedding(max_len, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_head, d_ff, drop_prob) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.token_emb(x) # [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        x += self.pos_emb(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)
        return x