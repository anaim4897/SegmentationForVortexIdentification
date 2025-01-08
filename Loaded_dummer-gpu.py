# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, jaccard_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

velocity_data = torch.load('kolflow_nu_0p0045_5k_256x256.pt.pt.pt').to(device)
velocity_data = torch.tensor(velocity_data).to(device)
vorticity_mask = torch.load('new_vorticity_mask.pt').to(device)


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


batch_size = 25
num_epochs = 10

def calculate_metrics(predictions, ground_truth, average_type='binary'):
    """
    Calculate and return common classification metrics.

    Args:
    predictions (numpy.array): The predictions from the model, should be binary or class indices for classification.
    ground_truth (numpy.array): The actual labels.
    average_type (str): Type of averaging performed on the data, applicable for multi-class classification.

    Returns:
    dict: Dictionary containing metric names and their corresponding values.
    """
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

# //////////////////////////////////////////training//////////////////////////////////////////////////////


from transformers import SegformerForSemanticSegmentation, SegformerConfig

config = SegformerConfig()  # Directly specify the number of labels
model = SegformerForSemanticSegmentation(config=config)

# Adjust the first convolutional layer to accept 2 channels instead of 3
model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(2, 32, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3),
                                                             bias=False)
# Assuming you have access to modify the model's segmentation head:
model.decode_head.classifier = nn.Conv2d(in_channels=model.decode_head.classifier.in_channels,
                                         out_channels=1, kernel_size=(1, 1))

# Move the model to GPU
model = model.to(device)
model.load_state_dict(torch.load('SegParamaters.pth'))
model.eval()  # Set the model to evaluation mode

# Define your optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define your loss function
criterion = nn.BCEWithLogitsLoss()

eval_velocity_data = velocity_data[4500:5000].to(device)
eval_high_vorticity_mask = vorticity_mask[4500:5000].to(device)

# Assuming you have a separate dataset for evaluation
eval_dataset = VortexDataset(eval_velocity_data, eval_high_vorticity_mask)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

total_correct = 0
total_samples = 0
counter = 1

# Create a single figure object
gif = plt.figure(figsize=(30, 10))
giff = plt.figure(figsize=(30, 10))
eval_metrics = []

with torch.no_grad():
    for batch in eval_dataloader:
        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)
        num_loop = int((len(eval_velocity_data) / batch_size)).__int__() # 20
        columns = int(num_loop / 2).__int__()  # 10

        outputs = model(inputs)['logits']

        # Upsample outputs to match mask size
        outputs_upsampled = F.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)

        predictions = (outputs_upsampled > 0.35).int()  # Convert to binary predictions

        '''with torch.no_grad():
            # Calculate metrics for the current batch
            batch_metrics = calculate_metrics(predictions, labels)
            eval_metrics.append(batch_metrics)'''

        total_correct += (predictions == labels).sum().item()
        total_samples += labels.numel()

        if counter % 4 == 0:
            # Create subplots within the single figure
            ax1 = gif.add_subplot(1, 5, int(counter/4))
            ax2 = giff.add_subplot(1, 5, int(counter/4))

            # Plot the labels and predictions
            ax1.set_title(f"label {counter/4}")
            im1 = ax1.imshow(labels.squeeze(1)[0].cpu())
            'plt.colorbar(im1, ax=ax1)'

            ax2.set_title(f"pred {counter/4}")
            im2 = ax2.imshow(predictions.squeeze(1)[0].cpu())
            'plt.colorbar(im2, ax=ax2)'

        counter += 1

    '''plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)'''
    'gif.tight_layout()  # This adjusts the plot spacing to prevent overlap'

    # Save the entire figure containing all subplots
    gif.savefig("seg_eval_plots_label.png")  # Save the figure in the current working directory
    giff.savefig("seg_eval_plots_pred.png")  # Save the figure in the current working directory

    # Close the figure to release memory
    plt.close(gif)
    plt.close(giff)

accuracy = total_correct / total_samples
print(f"Accuracy on evaluation data: {accuracy * 100:.2f}%")


'''
with open('seg_eval_metrics.txt', 'w') as file:
    # Write each element of the list to the file
    for item in eval_metrics:
        file.write(str(item) + '\n')  # Write each item followed by a newline character
'''