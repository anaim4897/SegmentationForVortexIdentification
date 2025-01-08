import matplotlib.pyplot as plt
import ast

import numpy as np

# Path to the metrics file
file_path = '/Users/alinaim/Downloads/vort/0.35/seg_train_metrics.txt'

# Read and parse the metrics file
metrics_data = []
with open(file_path, 'r') as file:
    for line in file:
        # Convert the single-quote dictionary string to a dictionary
        data = ast.literal_eval(line.strip())
        metrics_data.append(data)

jaccard_scores = [batch['average_jaccard_score'] for batch in metrics_data][::10]
precision_scores = [batch['average_precision'] for batch in metrics_data][::10]
recall_scores = [batch['average_recall'] for batch in metrics_data][::10]
f1_scores = [batch['average_f1_score'] for batch in metrics_data][::10]

# Convert lists to numpy arrays
jaccard_array = np.array(jaccard_scores)
precision_array = np.array(precision_scores)
recall_array = np.array(recall_scores)
f1_array = np.array(f1_scores)

'''
# Use numpy's mean function
mean_jaccard_np = np.mean(jaccard_array)
mean_precision_np = np.mean(precision_array)
mean_recall_np = np.mean(recall_array)
mean_f1_np = np.mean(f1_array)

print("Mean Jaccard Score (NumPy):", mean_jaccard_np)
print("Mean Precision (NumPy):", mean_precision_np)
print("Mean Recall (NumPy):", mean_recall_np)
print("Mean F1 Score (NumPy):", mean_f1_np)
'''

highest_jaccard_score = max(jaccard_scores)
highest_precision = max(precision_scores)
highest_recall = max(recall_scores)
highest_f1_score = max(f1_scores)

print("Highest Jaccard Score:", highest_jaccard_score)
print("Highest Precision:", highest_precision)
print("Highest Recall:", highest_recall)
print("Highest F1 Score:", highest_f1_score)

'''
index_highest_jaccard = jaccard_scores.index(highest_jaccard_score)
index_highest_precision = precision_scores.index(highest_precision)
index_highest_recall = recall_scores.index(highest_recall)
index_highest_f1 = f1_scores.index(highest_f1_score)

print("Batch with Highest Jaccard Score:", metrics_data[index_highest_jaccard])
print("Batch with Highest Precision:", metrics_data[index_highest_precision])
print("Batch with Highest Recall:", metrics_data[index_highest_recall])
print("Batch with Highest F1 Score:", metrics_data[index_highest_f1])
'''

# Prepare the x-axis values since we are plotting every 5th point
x_values = range(0, len(metrics_data), 10)  # Adjust range based on the number of batches and step

plt.figure(figsize=(12, 8))

# Plot Jaccard Score
plt.subplot(2, 2, 1)
plt.plot(x_values, jaccard_scores, label='Jaccard Score', marker='o', linestyle='-', color='b')
plt.title('Jaccard Scores Over Batches')
plt.xlabel('Batch Number')
plt.ylabel('Jaccard Score')
plt.grid(True)
plt.legend()

# Plot Precision
plt.subplot(2, 2, 2)
plt.plot(x_values, precision_scores, label='Precision', marker='o', linestyle='-', color='r')
plt.title('Precision Scores Over Batches')
plt.xlabel('Batch Number')
plt.ylabel('Precision')
plt.grid(True)
plt.legend()

# Plot Recall
plt.subplot(2, 2, 3)
plt.plot(x_values, recall_scores, label='Recall', marker='o', linestyle='-', color='g')
plt.title('Recall Scores Over Batches')
plt.xlabel('Batch Number')
plt.ylabel('Recall')
plt.grid(True)
plt.legend()

# Plot F1 Score
plt.subplot(2, 2, 4)
plt.plot(x_values, f1_scores, label='F1 Score', marker='o', linestyle='-', color='c')
plt.title('F1 Scores Over Batches')
plt.xlabel('Batch Number')
plt.ylabel('F1 Score')
plt.grid(True)
plt.legend()

# Display the plot
plt.tight_layout()
'''plt.savefig('/Users/alinaim/Downloads/vort/0.35/seg_train_metrics.png')
'''
plt.show()