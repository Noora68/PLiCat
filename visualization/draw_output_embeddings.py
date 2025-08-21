import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Union, Tuple, Optional
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
import matplotlib
from collections import defaultdict

# Find the project root directory and call os.path.dirname multiple times to find the directory upwards
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from PLiCat.PLiCat_dataset import SequenceDataset, collate_fn_dynamic_padding
from PLiCat.PLiCat_model import PLiCat
from PLiCat.utils import load_model_from_checkpoint

device = "cuda" if torch.cuda.is_available() else "cpu" 

# --------------------------
# 1. Data Loading and Preprocessing
# --------------------------
test_path = './data/test_dataset.csv'
df = pd.read_csv(test_path)
raw_labels = df['lipid_Lipidmaps_categories'].tolist()  # Raw label strings

BATCH_SIZE = 32
num_workers = 0  

# Create DataLoader
sequence_dataset = SequenceDataset(test_path)
sequence_loader = DataLoader(sequence_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn_dynamic_padding, shuffle=False) 

# --------------------------
# 2. Label Mapping 
# --------------------------
# Define target label dictionary
label_dict = {
    "0": "Fatty Acyl (FA)",
    "1": "Prenol Lipid (PR)",
    "2": "Polyketide (PK)",
    "3": "Sterol Lipid (ST)",
    "4": "Glycerophospholipid (GP)",
    "5": "Fatty Acyl (FA), Glycerophospholipid (GP)",
    "6": "NotLipidType",
    "7": "Prenol Lipid (PR), Glycerophospholipid (GP)",
    "8": "Fatty Acyl (FA), Prenol Lipid (PR)",
    "9": "Fatty Acyl (FA), Glycerolipid (GL)",
    "10": "Glycerolipid (GL)",
    "11": "Fatty Acyl (FA), Sterol Lipid (ST)",
    "12": "Fatty Acyl (FA), Prenol Lipid (PR), Glycerophospholipid (GP)",
    "13": "Fatty Acyl (FA), Polyketide (PK)",
    "14": "Fatty Acyl (FA), Sphingolipid (SP)",
    "15": "Prenol Lipid (PR), Glycerophospholipid (GP), Glycerolipid (GL)",
    "16": "Prenol Lipid (PR), Glycerolipid (GL)",
    "17": "Saccharolipid (SL)",
    "18": "Sphingolipid (SP)",
    "19": "Prenol Lipid (PR), Polyketide (PK)",
    "20": "Glycerophospholipid (GP), Sterol Lipid (ST)",
    "21": "Fatty Acyl (FA), Glycerophospholipid (GP), Sterol Lipid (ST)",
    "22": "Prenol Lipid (PR), Sterol Lipid (ST)"
}

# Build mapping: label name -> integer ID (for converting raw_labels to integers)
name_to_id = {v: int(k) for k, v in label_dict.items()}

# Convert raw label strings to integer IDs (matching color_list indices)
labels = []
for label in raw_labels:
    if label in name_to_id:
        labels.append(name_to_id[label])
    else:
        print(f"Warning: Label '{label}' not found in label_dict. Assigning default ID.")
        labels.append(0)  # Default to first category

labels_np = np.array(labels)  # Convert to numpy array for plotting

# --------------------------
# 3. Color Mapping and UMAP Dimensionality Reduction
# --------------------------
# Define color list (matches label IDs)
color_list = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#ffe119",  # yellow
    "#0082c8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#46f0f0",  # cyan
    "#f032e6",  # pink
    "#d2f53c",  # lime green
    "#fabebe",  # light pink
    "#008080",  # teal
    "#e6beff",  # lavender
    "#aa6e28",  # brown
    "#fffac8",  # beige
    "#800000",  # maroon
    "#aaffc3",  # mint green
    "#808000",  # olive
    "#ffd8b1",  # apricot
    "#000080",  # navy
    "#808080",  # gray
    "#e7298a",  # white (changed to magenta for visibility)
    "#000000",  # black
    "#a9a9a9",  # dark gray
]

outpath = './PLiCat_embedding_result'
os.makedirs(outpath, exist_ok=True)

# Replace "your_model.pt" ith the actual model
model = load_model_from_checkpoint(PLiCat, "your_model.pt", device=device)

# Switch to evaluation mode
model.eval()
model.zero_grad()

# --------------------------
# 4. Model Inference 
# --------------------------
print('Starting inference...')

with torch.no_grad():
    all_embeddings = []

    for batch in tqdm(sequence_loader, desc="Processing batches"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Get model outputs
        _, batch_embeddings = model(input_ids, attention_mask)
        all_embeddings.append(batch_embeddings.cpu())

    # Concatenate all batch embeddings
    protein_embeddings = torch.cat(all_embeddings, dim=0)

# Convert embeddings to numpy array (UMAP requires numpy format)
protein_embeddings_np = protein_embeddings.numpy()
   
# UMAP dimensionality reduction
print("Running UMAP dimensionality reduction...")
umap_model = UMAP(n_components=2, random_state=42, n_jobs=-1)  # Use all available cores
emb_2d = umap_model.fit_transform(protein_embeddings_np)

# --------------------------
# 5. Plotting (label-color matching)
# --------------------------
print("Generating visualization plot...")

plt.figure(figsize=(12, 6), dpi=300)

# Track which labels we've already added to legend
handled_labels = set()

# Plot by category (using integer label IDs)
for label_id in np.unique(labels_np):
    if label_id >= len(color_list):
        print(f"Warning: label_id {label_id} out of range, defaulting to gray")
        color = "#808080"
    else:
        color = color_list[label_id]

    indices = labels_np == label_id
    label_name = label_dict.get(str(label_id), f"Unknown_{label_id}")
    
    # Only add to legend if we haven't seen this label
    if label_id not in handled_labels:
        plt.scatter(
            emb_2d[indices, 0],
            emb_2d[indices, 1],
            label=label_name,
            s=15,
            alpha=0.7,
            color=color
        )
        handled_labels.add(label_id)
    else:
        plt.scatter(
            emb_2d[indices, 0],
            emb_2d[indices, 1],
            s=15,
            alpha=0.7,
            color=color
        )

# Add legend and title
plt.legend(title="Lipid Categories", fontsize=8, title_fontsize=10, 
           loc='best', bbox_to_anchor=(1.05, 1))
plt.title("UMAP Projection of PLiCat-model Embeddings on test_dataset", fontsize=16)
plt.xlabel("UMAP Dimension 1", fontsize=12)
plt.ylabel("UMAP Dimension 2", fontsize=12)
plt.grid(alpha=0.1)

# Save image
output_path = os.path.join(outpath, "trained_PLiCat-model_embeddings_umap.png")
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Visualization saved to {output_path}")
