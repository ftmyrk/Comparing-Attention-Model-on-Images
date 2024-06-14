import torch
import torch.nn as nn

class SparseAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout=0.1, sparsity_factor=0.5):
        super(SparseAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = nn.Dropout(dropout)
        self.sparsity_factor = sparsity_factor
        self.qkv_linear = nn.Linear(d_model, d_k + d_v)
        self.out_linear = nn.Linear(d_v, d_model)
    
    def forward(self, x):
        qkv = self.qkv_linear(x)
        q, k, v = qkv.split([self.d_k, self.d_k, self.d_v], dim=-1)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Apply sparsity
        topk = int(self.sparsity_factor * attn_weights.size(-1))
        topk_weights, _ = torch.topk(attn_weights, topk, dim=-1)
        sparse_mask = attn_weights >= topk_weights[:, :, -1:]
        attn_weights = attn_weights * sparse_mask.float()
        
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return self.out_linear(output)
