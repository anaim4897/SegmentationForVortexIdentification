import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load velocity data onto GPU
velocity_data = torch.load('kolflow_nu_0p0045_5k_256x256.pt.pt.pt').to(device)

# Shape: [batch_size, channels, height, width]
# Select a single dataset from one batch and one channel
selected_data_v = velocity_data[:, 1, :, :].to(device)  # Adjust the indices as needed
selected_data_u = velocity_data[:, 0, :, :].to(device)  # Adjust the indices as needed

# Compute second-order central differences using torch.gradient
grad_x, = torch.gradient(selected_data_v, dim=1)  # Compute du/dx along the width dimension
grad_y, = torch.gradient(selected_data_u, dim=1)  # Compute du/dx along the width dimension

vorticity = grad_x - grad_y
absvorticity = abs(vorticity)

# Assuming 'vorticity' is a NumPy array with computed vorticity values
threshold = 0.32  # Define your vorticity threshold
vorticity_mask = (absvorticity > threshold).float().to(device)

# Expand dimensions to match the model's output shape
'vorticity_mask = vorticity_mask.unsqueeze(1)'


class VortexDataset(Dataset):
    def __init__(self, velocity_data, vorticity_mask):
        self.velocity_data = velocity_data
        self.vorticity_mask = vorticity_mask

    def __len__(self):
        return len(self.velocity_data)

    def __getitem__(self, idx):
        sample = {
            'input': self.velocity_data[idx],
            'label': self.vorticity_mask[idx]
        }
        return sample


# Create an instance of the custom dataset
vortex_dataset = VortexDataset(velocity_data[0:4000], vorticity_mask[0:4000])

# Example usage of DataLoader for batching and shuffling
batch_size = 25
'32'
dataloader = DataLoader(vortex_dataset, batch_size=batch_size, shuffle=True)


class Segformer(nn.Module):
    def __init__(self, num_classes, image_size, patch_size, num_layers, embed_dim, num_heads, mlp_ratio=4,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm):
        super(Segformer, self).__init__()

        self.num_classes = num_classes
        self.patch_size = patch_size

        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(2, embed_dim, kernel_size=patch_size, stride=patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.dropout = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer
            ) for _ in range(num_layers)
        ])

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, image_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embed
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Only return the segmentation output, ignoring the class token
        x = x[:, 1:]
        x = self.head(x)
        x = x.unsqueeze(0)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor



# Define model parameters
num_classes = 1  # Number of segmentation classes
image_size = 256  # Input image size (square)
patch_size = 16  # Patch size
num_layers = 12  # Number of layers in the Segformer model
embed_dim = 384  # Embedding dimension
num_heads = 6  # Number of attention heads

# Instantiate the Segformer model
model = Segformer(
    num_classes=num_classes,
    image_size=image_size,
    patch_size=patch_size,
    num_layers=num_layers,
    embed_dim=embed_dim,
    num_heads=num_heads
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 10
'10'  # Adjust based on your training requirements

# Create a single figure object
fig = plt.figure(figsize=(10, 5))

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0.0

    for batch in dataloader:
        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels.unsqueeze(0))  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")

    # Create subplots within the single figure
    ax1 = fig.add_subplot(2, num_epochs, epoch + 1)
    ax2 = fig.add_subplot(2, num_epochs, epoch + num_epochs + 1)

    # Plot the labels and predictions
    ax1.set_title(f"label {epoch + 1} ({torch.where(velocity_data==labels[0])[0]})")
    im1 = ax1.imshow(labels.squeeze(1)[0].cpu())
    plt.colorbar(im1,ax=ax1)

    ax2.set_title(f"prediction {epoch + 1} ({torch.where(velocity_data==outputs[0])[0]})")
    im2 = ax2.imshow(outputs.detach().cpu().numpy().squeeze(1)[0])
    plt.colorbar(im2,ax=ax2)

# Save the entire figure containing all subplots
fig.savefig("new_trans_training_plots_combined.png")  # Save the figure in the current working directory

# Close the figure to release memory
plt.close(fig)

eval_velocity_data = velocity_data[4000:5000].to(device)
eval_high_vorticity_mask = vorticity_mask[4000:5000].to(device)

# Assuming you have a separate dataset for evaluation
eval_dataset = VortexDataset(eval_velocity_data, eval_high_vorticity_mask)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

model.eval()  # Set the model to evaluation mode
total_correct = 0
total_samples = 0
counter = 1

# Create a single figure object
gif = plt.figure(figsize=(10, 5))

with torch.no_grad():
    for batch in eval_dataloader:
        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)

        outputs = model(inputs)  # Forward pass
        predictions = (outputs > 0.5).float()  # Convert to binary predictions

        # Create subplots within the single figure
        ax1 = gif.add_subplot(2, num_epochs, counter)
        ax2 = gif.add_subplot(2, num_epochs, counter + num_epochs)

        # Plot the labels and predictions
        ax1.set_title(f"label {counter} ({torch.where(velocity_data==labels[0])[0]})")
        im1 = ax1.imshow(labels.squeeze(1)[0].cpu())
        plt.colorbar(im1,ax=ax1)

        ax2.set_title(f"prediction {counter} ({torch.where(velocity_data==predictions[0])[0]})")
        im2 = ax2.imshow(predictions.squeeze(1)[0].cpu())
        plt.colorbar(im2,ax=ax2)

        counter += 1
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.numel()



        # Save the entire figure containing all subplots
    gif.savefig("new_trans_eval_plots_combined.png")  # Save the figure in the current working directory

    # Close the figure to release memory
    plt.close(gif)

accuracy = total_correct / total_samples
print(f"Accuracy on evaluation data: {accuracy * 100:.2f}%")