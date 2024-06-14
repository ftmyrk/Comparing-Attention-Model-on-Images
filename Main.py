import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from MultiHeadAttention import MultiHeadAttention, PositionwiseFeedForward 
from SelfAttention import SelfAttention
from SparseAttention import SparseAttention
from AdditiveAttention import AdditiveAttention
from DynamicMaskAttention import DynamicMaskAttention
from CrossAttention import CrossAttention


# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, d_ff, dropout=0.1, attention_type='multihead'):
        super(TransformerBlock, self).__init__()
        if attention_type == 'multihead':
            self.attention = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout)
        elif attention_type == 'self':
            self.attention = SelfAttention(d_model, d_k, d_v, dropout)
        elif attention_type == 'sparse':
            self.attention = SparseAttention(d_model, d_k, d_v, dropout)
        elif attention_type == 'additive':
            self.attention = AdditiveAttention(d_model, d_k, d_v, dropout)
        elif attention_type == 'cross':
            self.attention = CrossAttention(d_model, d_k, d_v, dropout)
        elif attention_type == 'dynamic_mask':
            self.attention = DynamicMaskAttention(d_model, d_k, d_v, dropout)
        else:
            raise ValueError("Unsupported attention type")

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        normed_x = self.layer_norm1(x)
        q, k, v = normed_x, normed_x, normed_x  # Using the same input for q, k, v
        x = x + self.attention(q, k, v, mask)[0]  # Unpack the output tuple from attention
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

# Parameters
img_size = 32
patch_size = 4
num_classes = 100
d_model = 64
n_head = 8
d_k = d_v = 64
d_ff = 256
num_layers = 6
dropout = 0.1
attention_types = ['additive', 'cross', 'dynamic_mask', 'multihead', 'self', 'sparse', ]


# Instantiate models with different attention mechanisms
models = {att_type: VisionTransformer(img_size, patch_size, num_classes, d_model, n_head, d_k, d_v, d_ff, num_layers, dropout, att_type)
          for att_type in attention_types}



# Move models to CUDA
models = {att_type: model.to(device) for att_type, model in models.items()}


# Data preparation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    return epoch_losses

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move inputs and labels to the GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Instantiate models, criteria, and optimizers
criteria = {att_type: nn.CrossEntropyLoss().to(device) for att_type in attention_types}
optimizers = {att_type: optim.Adam(models[att_type].parameters(), lr=0.001) for att_type in attention_types}

# Store losses and accuracies for plotting
all_epoch_losses = {}

# Train and evaluate each model
for att_type in attention_types:
    print(f"Training {att_type} attention model...")
    epoch_losses = train_model(models[att_type], train_loader, criteria[att_type], optimizers[att_type], num_epochs=10)
    all_epoch_losses[att_type] = epoch_losses


    accuracy = evaluate_model(models[att_type], test_loader)
    print(f"Accuracy: {accuracy:.2f}%")
    

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot losses
# plt.subplot(1, 2, 1)
for att_type in attention_types:
    plt.plot(all_epoch_losses[att_type], label=f'{att_type} Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()
