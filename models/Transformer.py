from enc_dec.TransformerEncoder import TransformerEncoder
from enc_dec.TransformerDecoder import TransformerDecoder
from torch import nn

class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, n_head, d_ff,
                 n_layers, max_len, drop_prob=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(
            src_vocab, d_model, d_ff, n_head, n_layers, max_len, drop_prob
        )
        self.decoder = TransformerDecoder(
            tgt_vocab, d_model, max_len, n_head, d_ff, n_layers, drop_prob
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encoder(src)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        return self.fc_out(dec_out)