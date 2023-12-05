import os
import numpy as np
import torch
from datasets import load_dataset, Features, ClassLabel, Image
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import seaborn as sns
from sklearn.metrics import confusion_matrix

m = 'StreetCLIP-roberta-finetuned'

model = AutoModel.from_pretrained(m)
processor = AutoProcessor.from_pretrained(m)

# Define your class labels
choices = ['Chicago', 'London', 'Los Angeles', 'Melbourne', 'Miami', 'New York City', 'San Francisco', 'Singapore', 'Sydney', 'Toronto']

# Define your dataset directory
dataset_directory = 'data/raw_dataset'

# Define your metrics directory
metrics_directory = 'runs/streetclip_12_10_eval'

# Create metrics directory
os.makedirs(metrics_directory, exist_ok=True)

# Load your dataset
print('Loading dataset...')
dataset = load_dataset(dataset_directory, num_proc=8)
print('Dataset loaded!')

# Split dataset into train and test based on npy files
# Load train and test indices
train_indices = np.load('./idx/train_index.npy')
test_indices = np.load('./idx/test_index.npy')

# Create train and test datasets
print('Creating test/train dataset...')
train_dataset = dataset['train'].select(train_indices) # type: ignore
test_dataset = dataset['train'].select(test_indices) # type: ignore
print('Created test/train dataset!')

# Assuming each class has 20,000 samples in the original test dataset
samples_per_class = 20000
num_classes = 10
samples_per_class_reduced = 100  # Number of samples per class in the reduced dataset

# Calculate indices for each class
indices = []
for class_id in range(num_classes):
    start_idx = class_id * samples_per_class
    end_idx = start_idx + samples_per_class
    class_indices = np.arange(start_idx, end_idx)

    # Randomly sample from this class's indices
    np.random.shuffle(class_indices)
    selected_indices = class_indices[:samples_per_class_reduced]
    indices.extend(selected_indices)

# Now, indices contains the randomly selected indices for the reduced dataset
test_dataset_reduced = test_dataset.select(indices)

# Assuming you have set 'device' earlier in your code as follows:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device) # type: ignore

# Lists to hold all predictions and true labels
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    # Loop through the test_dataset
    for sample in tqdm(test_dataset_reduced, desc="Processing images"): # Insert the dataset you want to evaluate
        # Inputs processing
        inputs = processor(text=choices, images=sample['image'], return_tensors="pt", padding=True) # type: ignore
        # Move inputs to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        outputs = model(**inputs) # type: ignore
        
        # Logits and probabilities
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        all_probs.append(probs.cpu().numpy())

        # Predictions
        predicted_label_idx = probs.argmax().item()
        all_preds.append(predicted_label_idx)
        all_labels.append(sample['label']) # type: ignore

# Metrics calculation
accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')

# Print metrics
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Write metrics to a text file
with open(f'{metrics_directory}/metrics.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy * 100:.2f}%\n')
    f.write(f'Precision: {precision:.2f}\n')
    f.write(f'Recall: {recall:.2f}\n')
    f.write(f'F1 Score: {f1:.2f}\n')

# Generate the confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Confusion Matrix
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=choices, yticklabels=choices) # type: ignore
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.savefig(f"{metrics_directory}/confusion_matrix.png")  # Saving the figure
