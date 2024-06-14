import torch
import torch.nn as nn

class SparseAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout=0.1):
        super(SparseAttention, self).__init__()
        self.d_k = d_k
        self.w_qs = nn.Linear(d_model, d_k, bias=False)
        self.w_ks = nn.Linear(d_model, d_k, bias=False)
        self.w_vs = nn.Linear(d_model, d_v, bias=False)
        self.fc = nn.Linear(d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, q, k, v, mask=None):
        residual = q
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        sparsity_threshold = 0.9  # Example threshold, adjust as needed
        attn[attn < sparsity_threshold] = 0
        
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        output = self.fc(output)
        output += residual
        output = self.layer_norm(output)
        return output, attn
