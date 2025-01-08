import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import matplotlib.pyplot as plt

# move kolflow data onto csf using rsync
# then replace the file path with the one on csf
velocity_data = torch.load('/Users/alinaim/Downloads/kolflow_nu_0p0045_5k_256x256.pt.pt.pt')

# Shape: [batch_size, channels, height, width]
# Select a single dataset from one batch and one channel
'selected_data_u, selected_data_v = velocity_data[:, (0, 1), :, :]  # Adjust the indices as needed'
selected_data_v = velocity_data[:, 1, :, :]  # Adjust the indices as needed
selected_data_u = velocity_data[:, 0, :, :]  # Adjust the indices as needed

# Compute second-order central differences using torch.gradient
grad_x, = torch.gradient(selected_data_v, dim=1)  # Compute du/dx along the width dimension
grad_y, = torch.gradient(selected_data_u, dim=1)  # Compute du/dx along the width dimension

vorticity = grad_x - grad_y
absvorticity = abs(vorticity)

# Assuming 'vorticity' is a NumPy array with computed vorticity values
threshold = 0.23  # Define your vorticity threshold
vorticity_mask = (absvorticity > threshold).float()

# Expand dimensions to match the model's output shape
vorticity_mask = vorticity_mask.unsqueeze(1)

# Repeat the mask along the channel dimension to match the output shape
'vorticity_mask = vorticity_mask.repeat(1, 1, 256, 256)'


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
vortex_dataset = VortexDataset(velocity_data[0:100], vorticity_mask[0:100])

# Example usage of DataLoader for batching and shuffling
batch_size = 25
'32'
dataloader = DataLoader(vortex_dataset, batch_size=batch_size, shuffle=True)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels, out_channels, features=[32, 64, 128, 256],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


# Instantiate the model
input_channels = 2  # Assuming grayscale images as input
hidden_channels = batch_size  # SAME AS BATCH_SIZE Adjust based on the complexity of your task
output_channels = 1  # Assuming binary classification for vortex identification
input_size = 256  # Assuming square input images

model = UNET(input_channels, output_channels)

# Choose an optimizer (e.g., Adam) and a loss function (e.g., BCEWithLogitsLoss)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

num_epochs = 10
'10'  # Adjust based on your training requirements

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0.0

    for batch in dataloader:
        inputs = batch['input']
        labels = batch['label']

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels.float())  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")


'''
    plt.subplot(1, 2, 1)
    plt.title(f"label {epoch + 1}")
    plt.imshow(labels.squeeze(1)[0])
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title(f"prediction {epoch + 1}")
    plt.imshow(outputs.detach().numpy().squeeze(1)[0])
    plt.colorbar()

    plt.show()'''


eval_velocity_data = velocity_data[100:125]
eval_high_vorticity_mask = vorticity_mask[100:125]

# Assuming you have a separate dataset for evaluation
eval_dataset = VortexDataset(eval_velocity_data, eval_high_vorticity_mask)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

model.eval()  # Set the model to evaluation mode
total_correct = 0
total_samples = 0
counter = 1

with torch.no_grad():
    for batch in eval_dataloader:
        inputs = batch['input']
        labels = batch['label']

        outputs = model(inputs)  # Forward pass
        predictions = (outputs > 0.5).float()  # Convert to binary predictions

        plt.subplot(1, 2, 1)
        plt.title(f"label {counter}")
        plt.imshow(labels.squeeze(1)[0])
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title(f"prediction {counter}")
        plt.imshow(predictions.squeeze(1)[0])
        plt.colorbar()

        plt.show()

        counter += 1
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.numel()

accuracy = total_correct / total_samples
print(f"Accuracy on evaluation data: {accuracy * 100:.2f}%")

