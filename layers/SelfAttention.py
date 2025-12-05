import torch
from torch import nn

class Attention(nn.Module):
    """
    Scaled Dot-Product Attention,
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, q, k, v, mask=None):
        d_k = k.shape[-1]
        score = torch.matmul(q, k.transpose(-1, -2)) / (d_k ** 0.5)
        if mask is not None:
            score = score.masked_fill(mask==0, float('-inf'))
        score = self.softmax(score)
        return torch.matmul(score, v)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = Attention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        out = self.attention(q, k, v, mask)
        out = self.concat(out)

        return self.w_concat(out)

    def split(self, x):
        """
        split tensor into multi head
        :param x: [batch_size, seq_len, d_model]
        :return: [batch_size, n_head, seq_len, d_head]
        """
        batch_size, seq_len, d_model = x.shape
        d_head = d_model // self.n_head

        return x.view((batch_size, seq_len, self.n_head, d_head)).transpose(1,2)

    def concat(self, x):
        """
        concatenate multi head attention
        :param x: [batch_size, n_head, seq_len, d_head]
        :return: [batch_size, seq_len, d_model]
        """
        batch_size, n_head, seq_len, d_head = x.shape
        return x.transpose(1,2).contiguous().view((batch_size, seq_len, d_head*n_head))
