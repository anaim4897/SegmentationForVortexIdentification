import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

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


# Create an instance of the custom dataset
vortex_dataset = VortexDataset(velocity_data[0:4500], vorticity_mask[0:4500])

# DataLoader for batching and shuffling
batch_size = 25
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
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
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
            skip_connection = skip_connections[idx // 2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


def calculate_metrics(predictions, ground_truth, average_type='binary'):
    # Move predictions and labels to CPU and convert to numpy
    predictions = predictions.squeeze(1).cpu().numpy()
    ground_truth = ground_truth.squeeze(1).cpu().numpy()

    # Reshape from [25, 256, 256] to [25, 65536]
    predictions_flat = predictions.reshape(predictions.shape[0], -1)
    ground_truth_flat = ground_truth.reshape(ground_truth.shape[0], -1)

    # Initialize lists to hold metric calculations for each sample
    jaccard_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # Compute metrics for each sample
    for i in range(predictions.shape[0]):
        jaccard_scores.append(jaccard_score(ground_truth_flat[i], predictions_flat[i], average=average_type))
        precision_scores.append(precision_score(ground_truth_flat[i], predictions_flat[i], average=average_type))
        recall_scores.append(recall_score(ground_truth_flat[i], predictions_flat[i], average=average_type))
        f1_scores.append(f1_score(ground_truth_flat[i], predictions_flat[i], average=average_type))

    # Calculate average of each metric
    metrics = {
        'average_jaccard_score': np.mean(jaccard_scores),
        'average_precision': np.mean(precision_scores),
        'average_recall': np.mean(recall_scores),
        'average_f1_score': np.mean(f1_scores)
    }
    return metrics


# Instantiate the model
input_channels = 2  # 2 velocity channels images as input
hidden_channels = batch_size  # SAME AS BATCH_SIZE
output_channels = 1  # binary classification for vortex identification
input_size = 256  # square input images

model = UNET(input_channels, output_channels).to(device)

# Choose an optimizer and a loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

num_epochs = 10

# Create a single figure object
fig = plt.figure(figsize=(30, 10))
figg = plt.figure(figsize=(30, 10))

loss_list = []
train_metrics = []

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0.0

    for batch in dataloader:
        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        predictions = (outputs > 0.5).int()  # Convert to binary predictions

        with torch.no_grad():
            # Calculate metrics for the current batch
            batch_metrics = calculate_metrics(predictions, labels)
            train_metrics.append(batch_metrics)

        loss = criterion(outputs, labels.float())  # Compute the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        loss_list.append(loss.item())
        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")

    # Create subplots within the single figure
    ax1 = fig.add_subplot(1, num_epochs, epoch + 1)
    ax2 = figg.add_subplot(1, num_epochs, epoch + 1)

    # Plot the labels and predictions
    ax1.set_title(f"label {epoch + 1}")
    im1 = ax1.imshow(labels.squeeze(1)[0].cpu())

    ax2.set_title(f"pred {epoch + 1}")
    im2 = ax2.imshow(outputs.detach().cpu().numpy().squeeze(1)[0])

# Save the entire figure containing all subplots
fig.savefig("unet_training_plots_label.png")  # Save the figure in the current working directory
figg.savefig("unet_training_plots_pred.png")  # Save the figure in the current working directory

# Close the figure to release memory
plt.close(fig)
plt.close(figg)

print(len(loss_list))
with open('unetloss.txt', 'w') as file:
    # Write each element of the list to the file
    for item in loss_list:
        file.write(str(item) + '\n')  # Write each item followed by a newline character'''

with open('unet_train_metrics.txt', 'w') as file:
    # Write each element of the list to the file
    for item in train_metrics:
        file.write(str(item) + '\n')  # Write each item followed by a newline character

# Save the parameters
torch.save(model.state_dict(), 'paramaters.pth')


eval_velocity_data = velocity_data[45000:5000].to(device)
eval_high_vorticity_mask = vorticity_mask[4500:5000].to(device)

# Have a separate dataset for evaluation
eval_dataset = VortexDataset(eval_velocity_data, eval_high_vorticity_mask)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

model.eval()  # Set the model to evaluation mode
total_correct = 0
total_samples = 0
counter = 0

# Create a single figure object
gif = plt.figure(figsize=(30, 10))
giff = plt.figure(figsize=(30, 10))
eval_metrics = []

with torch.no_grad():
    for batch in eval_dataloader:
        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)
        num_loop = (len(eval_velocity_data) / batch_size).int() #40
        columns = num_loop/4 #10

        outputs = model(inputs)  # Forward pass
        predictions = (outputs > 0.5).int()  # Convert to binary predictions

        with torch.no_grad():
            # Calculate metrics for the current batch
            batch_metrics = calculate_metrics(predictions, labels)
            eval_metrics.append(batch_metrics)

        total_correct += (predictions == labels).sum().item()
        total_samples += labels.numel()

        if counter % 4 == 0:
            # Create subplots within the single figure
            ax1 = gif.add_subplot(1, 5, int(counter / 4))
            ax2 = giff.add_subplot(1, 5, int(counter / 4))

            # Plot the labels and predictions
            ax1.set_title(f"label {counter/4}")
            im1 = ax1.imshow(labels.squeeze(1)[0].cpu())

            ax2.set_title(f"prediction {counter/4}")
            im2 = ax2.imshow(predictions.squeeze(1)[0].cpu())

        counter += 1

    # Save the entire figure containing all subplots
    gif.savefig("unet_eval_plots_label.png")  # Save the figure in the current working directory
    giff.savefig("unet_eval_plots_pred.png")  # Save the figure in the current working directory

    # Close the figure to release memory
    plt.close(gif)
    plt.close(giff)


with open('unet_eval_metrics.txt', 'w') as file:
    # Write each element of the list to the file
    for item in eval_metrics:
        file.write(str(item) + '\n')  # Write each item followed by a newline character