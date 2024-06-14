import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.w_q = nn.Linear(d_model, d_k, bias=False)
        self.w_k = nn.Linear(d_model, d_k, bias=False)
        self.w_v = nn.Linear(d_model, d_v, bias=False)
        self.fc = nn.Linear(d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        residual = q
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, v)
        output = self.fc(output)
        output += residual
        output = self.layer_norm(output)

        return output, attn
