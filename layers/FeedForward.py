from torch import nn

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, drop_prob=0.1):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.ffn(x)