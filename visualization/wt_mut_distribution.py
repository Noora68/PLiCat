import os
import sys
import torch
from typing import List
from tqdm import tqdm
import numpy as np
import pandas as pd
from esm.tokenization import EsmSequenceTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA

# Find the project root directory and call os.path.dirname multiple times to find the directory upwards
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from PLiCat.PLiCat_dataset import DiseaseDataset
from PLiCat.PLiCat_model import PLiCat
from PLiCat.utils import binary_entropy, shannon_entropy_batch, shannon_entropy_per_label, load_model_from_checkpoint

# Inference and evaluation
if __name__ == "__main__":

    # Create output directory
    output_dir = 'disease-results'
    os.makedirs(output_dir, exist_ok=True)

    # Example sequences (comment out during actual run, use file data)
    # sequences = [
    #     "SILWHEMWHEGLEEASRLYFGERNVKGMFEVLEPLHAMMERGPQTLKETSFNQAYGRDLMEAQEWCRKYMKSGNVKDLTQAWDLYYHVFRRIS",
    #     "DNFTGTYKMWMFIDPRRALLFIASFQILLGILIHMIVLGSDLNWHSDGIPKFYFPNAAEASAPIDMSPIPSARNFKFD",
    #     "MVLSATTIGALLGLGTQMYSNALRKLPYMRHPWEHVVGMGLGAVFVNQLLKWEAQVEQDLDKMLEKAKAANERRYIDGDDD"
    # ]
    batch_size = 32
    threshold = 0.6
    test_path = './data/disease_merged_35_500_filtered.csv'  # Test file
    df = pd.read_csv(test_path)
    WT_Seqs = df['WT_Seq'].tolist()
    Mut_Seqs = df['Mut_Seq'].tolist()
    print(f"Wild-type sequence:\n{WT_Seqs[0]}\nvs Mutant sequence:\n{Mut_Seqs[0]}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Replace "your_model.pt" ith the actual model
    model = load_model_from_checkpoint(PLiCat, "your_model.pt", device=device)

    model = model.to(device)
    model.eval()  # Switch to evaluation mode
    model.zero_grad()  # Clear gradients (to avoid accumulation)

    types = ['WT_Seqs', 'Mut_Seqs']
    # 2. Create datasets and data loaders
    dataset = {}
    dataloader = {}

    dataset['WT_Seqs'] = DiseaseDataset(WT_Seqs)  # Import according to actual situation
    dataloader['WT_Seqs'] = DataLoader(dataset['WT_Seqs'], batch_size=batch_size, shuffle=False)
    dataset['Mut_Seqs'] = DiseaseDataset(Mut_Seqs)  # Import according to actual situation
    dataloader['Mut_Seqs'] = DataLoader(dataset['Mut_Seqs'], batch_size=batch_size, shuffle=False)

    entropy_dict = {}
    entropy_dict_per_label = {}
    # Store all features to be compared
    features = {
        'model_probs': {},
        'model_embeddings': {},
        'model_logits': {}
    }
    LABEL_NAMES = ['NO', 'FA', 'PR', 'GP', 'ST', 'PK', 'GL', 'SP', 'SL']
    # Define before loop
    num_labels = len(LABEL_NAMES)  # Assuming 9 labels
    tokenizer = EsmSequenceTokenizer()
    for type in types:
        print(f"\n {type} inference started...")

        # 3. Initialize storage lists
        model_probs = []  # Probability results of current model
        model_embeddings = []
        model_logits = []
        all_sequences = []
        with torch.no_grad():
            for batch_sequences in tqdm(dataloader[f'{type}'], desc="Processing batches",
                                        leave=False):  # Enable during actual run

                input_ids = [torch.tensor(tokenizer.encode(seq), dtype=torch.long) for seq in batch_sequences]

                # Pad to the longest sequence in the batch
                input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

                # Attention mask: 1 for real tokens, 0 for padding
                attention_mask = (input_ids_padded != tokenizer.pad_token_id).long()
                input_ids_padded = input_ids_padded.to(device)
                attention_mask = attention_mask.to(device)

                # 5. Model prediction (enable during actual run)
                # batch_logits, batch_embeddings = model(input_ids_padded, attention_mask)
                batch_logits, batch_embeddings = model(input_ids_padded, attention_mask)

                # 7. Calculate probabilities and store
                batch_probs = torch.sigmoid(batch_logits)
                model_probs.append(batch_probs.detach().cpu())  # Move back to CPU and detach from computation graph
                model_embeddings.append(batch_embeddings.detach().cpu())
                model_logits.append(batch_logits.detach().cpu())

                all_sequences.extend(batch_sequences)  # Enable during actual run

        # 8. Merge all batch results for current model
        model_probs = torch.cat(model_probs, dim=0)
        model_embeddings = torch.cat(model_embeddings, dim=0)
        model_logits = torch.cat(model_logits, dim=0)

        # Store features for subsequent comparison
        features['model_probs'][type] = model_probs.numpy()
        features['model_embeddings'][type] = model_embeddings.numpy()
        features['model_logits'][type] = model_logits.numpy()

        # 11. Apply threshold for binarized prediction
        threshold = 0.6
        model_preds = (model_probs >= threshold).int()

        # Save inference results
        np.save(os.path.join(output_dir, f"{type}_model_probs.npy"), model_probs.numpy())
        np.save(os.path.join(output_dir, f"{type}_model_preds.npy"), model_preds.numpy())
        np.save(os.path.join(output_dir, f"{type}_model_logits.npy"), model_logits.numpy())  # Fix typo in original code
        np.save(os.path.join(output_dir, f"{type}_model_embeddings.npy"),
                model_embeddings.numpy())  # Fix typo in original code

        # Output results
        print("Probabilities shape:", model_probs.shape)
        print("Predictions shape:", model_preds.shape)
        print("Model output logits shape:", model_logits.shape)
        print("Model output embeddings:", model_embeddings.shape)
        print("\nProbabilities model_probs[0]:")
        print(model_probs[0])
        print("\nPredictions model_preds[0]:")
        print(model_preds[0])
        print("\nModel Logits: model_logits[0]:")
        print(model_logits[0])
        print("\nModel embeddings model_embeddings[0]:")
        print(model_embeddings[0])

        print(f"\n Inference completed, starting evaluation...")

        # ============ Evaluation, calculate Shannon Entropy  ===============
        # Simulate Shannon entropy calculation (replace with actual calculation during run)
        # Statistics per sample
        entropy_values = shannon_entropy_batch(model_logits).cpu().numpy()
        entropy_dict[type] = entropy_values  # Save sample-level entropy for each category
        print(entropy_values.shape)

        plt.hist(entropy_values, bins=30)
        plt.xlabel("Shannon Entropy")
        plt.ylabel("Number of Samples")
        plt.title(f"Entropy({type})_per_sample Distribution of Model Predictions")
        plt.savefig(os.path.join(output_dir, f'Entropy({type})_per_sample.png'), dpi=300)
        plt.close()

        # Statistics per label
        entropy_values_per_label = shannon_entropy_per_label(model_logits).cpu().numpy()
        entropy_dict_per_label[type] = entropy_values_per_label  # Save label-level entropy for each category
        print(entropy_values_per_label.shape)

        plt.figure(figsize=(8, 4))
        plt.bar(LABEL_NAMES, entropy_values_per_label)
        plt.xlabel("Labels")
        plt.ylabel("Shannon Entropy")
        plt.title(f"Entropy({type}) per Label")
        plt.savefig(os.path.join(output_dir, f'Entropy({type})_per_label.png'), dpi=300)
        plt.close()  # Use close() instead of show() to avoid blocking

        # Add after loop ends
        torch.cuda.empty_cache()

    # --- Distribution distance comparison ---
    print("\n====================")
    print("Distribution distance comparison (WT vs MUT)")

    wt_entropy_per_label = entropy_dict_per_label['WT_Seqs']
    mut_entropy_per_label = entropy_dict_per_label['Mut_Seqs']

    # (1) Label-level bar comparison chart for Shannon entropy
    x = np.arange(len(LABEL_NAMES))
    width = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, wt_entropy_per_label, width, label='WT_Seq')
    plt.bar(x + width / 2, mut_entropy_per_label, width, label='Mut_Seq')

    plt.xlabel('Labels')
    plt.ylabel('Shannon Entropy')
    plt.title('Shannon Entropy Comparison between WT and Mut Sequences')
    plt.xticks(x, LABEL_NAMES)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Shannon_Entropy_Comparison.png'), dpi=300)
    plt.close()

    # Wasserstein Distance
    wasserstein_results = {}

    print(f"WT logits shape: {features['model_logits']['WT_Seqs'].shape}")
    print(f"Mut logits shape: {features['model_logits']['Mut_Seqs'].shape}")

    # Process embeddings with PCA
    pca = PCA(n_components=3)

    wt_emb_pca = pca.fit_transform(features['model_logits']['WT_Seqs'])
    mut_emb_pca = pca.transform(features['model_logits']['Mut_Seqs'])

    # Calculate explained variance ratio for each principal component
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Add 3D visualization
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot scatter plot
    scatter_wt = ax.scatter(wt_emb_pca[:, 0], wt_emb_pca[:, 1], wt_emb_pca[:, 2],
                            c='b', alpha=0.6, s=30, label='WT Sequences')
    scatter_mut = ax.scatter(mut_emb_pca[:, 0], mut_emb_pca[:, 1], mut_emb_pca[:, 2],
                             c='r', alpha=0.6, s=30, label='Mutant Sequences')

    # Set axis labels and title
    ax.set_xlabel(f'PC1 ({explained_variance[0] * 100:.1f}% Variance)', fontsize=15, labelpad=15)
    ax.set_ylabel(f'PC2 ({explained_variance[1] * 100:.1f}% Variance)', fontsize=15, labelpad=15)
    ax.set_zlabel(f'PC3 ({explained_variance[2] * 100:.1f}% Variance)', fontsize=15, labelpad=15)
    ax.set_title('3D Embedding Space Visualization (PCA Projection)', fontsize=16, pad=20)

    # Add legend
    ax.legend(fontsize=13, loc='upper right')

    # Add grid and adjust viewing angle
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.view_init(elev=25, azim=45)  # Set better viewing angle

    # Add explanatory text
    text_str = (
        "PCA Projection of Protein Sequence Embeddings\n"
        f"Total Explained Variance: {cumulative_variance[2] * 100:.1f}%\n"
        "PC1: Primary embedding direction capturing the most variation\n"
        "PC2: Secondary embedding direction orthogonal to PC1\n"
        "PC3: Tertiary embedding direction capturing additional structure"
    )
    plt.figtext(0.5, 0.01, text_str, ha='center', fontsize=12,
                bbox=dict(facecolor='lightyellow', alpha=0.5, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make space for bottom text
    plt.savefig(os.path.join(output_dir, '3D_Embedding_Space.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # (2) Wasserstein Distance comparison chart for model_logits
    wasserstein_logits = []
    for i in range(len(LABEL_NAMES)):
        wt_data = features['model_logits']['WT_Seqs'][:, i]
        mut_data = features['model_logits']['Mut_Seqs'][:, i]
        distance = wasserstein_distance(wt_data, mut_data)
        wasserstein_logits.append(distance)
    wasserstein_results['model_logits'] = wasserstein_logits

    plt.figure(figsize=(8, 4))
    plt.bar(LABEL_NAMES, wasserstein_logits)
    plt.xlabel('Labels')
    plt.ylabel('Wasserstein Distance')
    plt.title('Wasserstein Distance of model_logits between WT and Mut Sequences')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Wasserstein_Distance_logits.png'), dpi=300)
    plt.close()

    # (3) Wasserstein Distance comparison chart for model_probs
    wasserstein_probs = []
    for i in range(len(LABEL_NAMES)):
        wt_data = features['model_probs']['WT_Seqs'][:, i]
        mut_data = features['model_probs']['Mut_Seqs'][:, i]
        distance = wasserstein_distance(wt_data, mut_data)
        wasserstein_probs.append(distance)
    wasserstein_results['model_probs'] = wasserstein_probs

    plt.figure(figsize=(8, 4))
    plt.bar(LABEL_NAMES, wasserstein_probs)
    plt.xlabel('Labels')
    plt.ylabel('Wasserstein Distance')
    plt.title('Wasserstein Distance of model_probs between WT and Mut Sequences')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Wasserstein_Distance_probs.png'), dpi=300)
    plt.close()

    # (4) JS Divergence comparison chart for model_probs
    js_probs = []
    for i in range(num_labels):
        p = features['model_probs']['WT_Seqs'][:, i]
        q = features['model_probs']['Mut_Seqs'][:, i]

        # Create histogram distributions
        hist_p, bin_edges = np.histogram(p, bins=50, range=(0, 1), density=True)
        hist_q, _ = np.histogram(q, bins=bin_edges, density=True)

        # Normalize
        hist_p = hist_p / hist_p.sum()
        hist_q = hist_q / hist_q.sum()

        # Ensure non-zero values
        hist_p = np.clip(hist_p, 1e-10, 1)
        hist_q = np.clip(hist_q, 1e-10, 1)

        # Calculate JS divergence
        distance = jensenshannon(hist_p, hist_q)
        js_probs.append(distance)

    plt.figure(figsize=(8, 4))
    plt.bar(LABEL_NAMES, js_probs)
    plt.xlabel('Labels')
    plt.ylabel('JS Divergence')
    plt.title('JS Divergence of model output between WT and Mut Sequences')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'JS_Divergence_probs.png'), dpi=300)
    plt.close()
