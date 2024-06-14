import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = nn.Dropout(dropout)
        self.qkv_linear = nn.Linear(d_model, d_k + d_v)
        self.out_linear = nn.Linear(d_v, d_model)
    
    def forward(self, x):
        qkv = self.qkv_linear(x)
        q, k, v = qkv.split([self.d_k, self.d_k, self.d_v], dim=-1)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return self.out_linear(output)
