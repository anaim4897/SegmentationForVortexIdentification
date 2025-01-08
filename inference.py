import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load velocity data onto GPU
velocity_data = torch.load('kolflow_nu_0p0045_5k_256x256.pt.pt.pt').to(device)
vorticity_mask = torch.load('new_vorticity_mask.pt').to(device)

# Expand dimensions to match the model's output shape
vorticity_mask = vorticity_mask.unsqueeze(1)

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


# Example usage of DataLoader for batching and shuffling
batch_size = 25
'32'

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

model = UNET(input_channels, output_channels).to(device)
model.load_state_dict(torch.load('paramaters.pth'))
model.eval()
total_correct = 0
total_samples = 0
counter = 0

eval_velocity_data = velocity_data[4500:5000].to(device)
eval_high_vorticity_mask = vorticity_mask[4500:5000].to(device)

# Assuming you have a separate dataset for evaluation
eval_dataset = VortexDataset(eval_velocity_data, eval_high_vorticity_mask)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)


def measure_inference_time(model, data_loader, device):
    total_time = 0
    total_batches = 0

    with torch.no_grad():
        for inputs in data_loader:
            inputs = inputs['input'].to(device)

            # Start timer
            start_time = time.time()

            # Model inference
            outputs = model(inputs)

            # End timer
            end_time = time.time()

            # Calculate elapsed time
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            total_batches += 1

    average_time_per_batch = total_time / total_batches
    return average_time_per_batch

# Call the function to measure inference time
average_inference_time = measure_inference_time(model, eval_dataloader, device)
print(f"Average inference time per batch: {average_inference_time:.3f} seconds")

