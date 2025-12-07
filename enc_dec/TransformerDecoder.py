from torch import nn
from layers.Embed import TokenEmbedding, PositionEmbedding
from blocks.DecoderBlock import DecoderBlock

class TransformerDecoder(nn.Module):
    def __init__(self, voc_size, d_model, max_len, n_head, d_ff, n_layers, drop_prob=0.1):
        super().__init__()
        self.token_emb = TokenEmbedding(voc_size, d_model)
        self.pos_emb = PositionEmbedding(max_len, d_model)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_head, d_ff, drop_prob) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, enc_out, tgt_mask, memory_mask):
        x = self.token_emb(x)  # [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        x += self.pos_emb(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, memory_mask)
        return x