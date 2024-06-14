import torch
import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention, PositionwiseFeedForward 
from SelfAttention import SelfAttention
from SparseAttention import SparseAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_ff, dropout=0.1, attention_type='multihead'):
        super(TransformerBlock, self).__init__()
        if attention_type == 'multihead':
            self.attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        elif attention_type == 'self':
            self.attention = SelfAttention(d_model, d_k, d_v, dropout)
        elif attention_type == 'sparse':
            self.attention = SparseAttention(d_model, d_k, d_v, dropout)
        else:
            raise ValueError("Unsupported attention type")
        
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, d_model, n_head, d_k, d_v, d_ff, num_layers, dropout=0.1, attention_type='multihead'):
        super(VisionTransformer, self).__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_patches = (img_size // patch_size) ** 2
        
        self.embedding = nn.Linear(patch_size * patch_size * 3, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_k, d_v, d_ff, dropout, attention_type)
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, self.patch_size, H // self.patch_size, self.patch_size, W // self.patch_size)
        x = x.permute(0, 3, 5, 2, 4, 1).contiguous()
        x = x.view(B, self.num_patches, -1)
        x = self.embedding(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embedding
        
        for block in self.transformer_blocks:
            x = block(x)
        
        cls_token_final = x[:, 0]
        out = self.fc(cls_token_final)
        return out

# Example usage
img_size = 32
patch_size = 4
num_classes = 10
d_model = 64
n_head = 8
d_k = d_v = 64
d_ff = 256
num_layers = 6
dropout = 0.1
attention_types = ['multihead', 'self', 'sparse']

# Instantiate models with different attention mechanisms
models = {att_type: VisionTransformer(img_size, patch_size, num_classes, d_model, n_head, d_k, d_v, d_ff, num_layers, dropout, att_type)
          for att_type in attention_types}

# Example data
x = torch.randn(16, 3, 32, 32)

# Forward pass
outputs = {att_type: model(x) for att_type, model in models.items()}

# Print the output shapes to verify
for att_type, output in outputs.items():
    print(f"{att_type} attention output shape: {output.shape}")