import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import fdrcorrection
from collections import defaultdict
from typing import List
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import LayerIntegratedGradients
from esm.tokenization import EsmSequenceTokenizer

# Find the project root directory and call os.path.dirname multiple times to find the directory upwards
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from PLiCat.PLiCat_model import PLiCat
from PLiCat.PLiCat_dataset import SequenceDataset, collate_fn_dynamic_padding
from PLiCat.utils importload_model_from_checkpoint

# Amino acid to index mapping (must be consistent with the model)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA = {idx: aa for aa, idx in AA_TO_IDX.items()}

def amino_acid_index(aa: str):
    return AMINO_ACIDS.index(aa.upper())

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)

    return attributions

def construct_input_and_baseline(input_ids_list, tokenizer, pad_token_id, device):
    # Batch conversion
    token_list = [tokenizer.convert_ids_to_tokens(seq) for seq in input_ids_list]

    # Find the maximum sequence length
    max_len = max(len(seq) for seq in input_ids_list)

    # Pad sequences
    input_ids = [seq + [pad_token_id] * (max_len - len(seq)) for seq in input_ids_list]
    baseline_ids = [[0] + [pad_token_id] * (len(seq) - 2) + [2] for seq in input_ids_list]
    baseline_ids = [seq + [pad_token_id] * (max_len - len(seq)) for seq in baseline_ids]

    # Generate attention mask (1 for non-PAD, 0 for PAD)
    attention_mask = [[1 if token != pad_token_id else 0 for token in seq] for seq in input_ids]

    # Convert to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).to(device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)

    return input_ids, baseline_ids, attention_mask, token_list

# --------------------------
# 1. Calculate attribution scores
# --------------------------
def compute_attributions(model, data_loader, tokenizer, device='cuda'):
    model.to(device)
    model.eval()

    # Define forward_func: only output the logit you want to explain
    def forward_func(input_ids, attention_mask=None):
        logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        return logits  # shape: (batch, num_labels)

    # Select the layer to explain, such as the embedding layer
    layer = model.model.embed
    lig = LayerIntegratedGradients(forward_func, layer)
    attr_scores = defaultdict(list)

    special_tokens = {"<cls>", "<eos>", "<pad>", "<unk>"}  # Define model special tokens

    for batch in tqdm(data_loader, desc="Batch Attribution"):
        input_ids_list = batch['input_ids'].tolist()
        labels = batch['labels']
        sequences = batch['sequence']

        input_ids, baseline_ids, attention_mask, token_list = \
            construct_input_and_baseline(input_ids_list, tokenizer, tokenizer.pad_token_id, device)

        for j in range(batch['labels'].size(0)):
            for tag_idx in range(batch['labels'].size(1)):
                if labels[j, tag_idx] != 1:
                    continue

                attributions, delta = lig.attribute(
                    inputs=input_ids,
                    baselines=baseline_ids,
                    additional_forward_args=(attention_mask,),
                    target=tag_idx,  # Index of the label you want to explain, e.g., the 0th label
                    return_convergence_delta=True,
                    n_steps=20
                )

                attributions_sum = summarize_attributions(attributions)
                if attributions_sum.dim() == 1:
                    attributions_sum = attributions_sum.unsqueeze(0)  # Add a dimension at dim 0 to become (1, seq_len)
                # attributions_sum
                # tensor([[ 0.0000, -0.2651,  0.6626,  0.0294,  0.0000,  0.0000,  0.0000],
                #         [ 0.0000,  0.0981,  0.4465,  0.0000,  0.0000,  0.0000,  0.0000],
                #         [ 0.0000,  0.1406, -0.2774, -0.0559,  0.0070,  0.0000,  0.0000],
                #         [ 0.0000, -0.0128,  0.3020, -0.2349, -0.1748, -0.0629,  0.0000]],
                #        device='cuda:0')

                for pos, aa in enumerate(sequences[j].upper()):  # Contribution value of each amino acid in the current sequence to the current label
                    if aa in AMINO_ACIDS:
                        # attr_scores[(tag_idx, aa)].append(pos_attr[pos])
                        attr_scores[(tag_idx, aa)].append(attributions_sum[j, pos])

    print(f"Number of attribution samples (e.g., FA, A): {len(attr_scores.get((1, 'A'), []))}")
    if len(attr_scores.get((1, 'A'), [])) >= 2:
        print(f"Example: {attr_scores[(1, 'A')][0]},{attr_scores[(1, 'A')][1]}")

    return attr_scores

# --------------------------
# 2. Plot attribution heatmap
# --------------------------
def compute_ttest_and_heatmap(attr_scores, save_dir='./heatmap', prefix='attribution', cluster=True):
    os.makedirs(save_dir, exist_ok=True)

    basic_labels = ['NO', 'FA', 'PR', 'GP', 'ST', 'PK', 'GL', 'SP', 'SL']
    n_classes = len(basic_labels)

    heatmap_data = np.zeros((n_classes, len(AMINO_ACIDS)))
    pval_matrix = np.ones_like(heatmap_data)

    # Calculate the attribute values for the heatmap matrix
    for tag_idx in range(n_classes):
        for aa_idx, aa in enumerate(AMINO_ACIDS):
            key = (tag_idx, aa)  # There are 9 × 20 = 180 combinations in total
            target_scores = attr_scores.get(key, [])
            if len(target_scores) < 2:
                continue

            other_scores = []
            for other_tag in range(n_classes):
                if other_tag == tag_idx:
                    continue
                other_scores.extend(attr_scores.get((other_tag, aa), []))

            if len(other_scores) < 2:
                continue

            target_scores = [t.detach().cpu().numpy() for t in target_scores]
            other_scores = [t.detach().cpu().numpy() for t in other_scores]

            t_stat, p_val = ttest_ind(target_scores,
                                      other_scores,
                                      equal_var=False)
            heatmap_data[tag_idx, aa_idx] = np.mean(target_scores)  # Attribution value
            pval_matrix[tag_idx, aa_idx] = p_val  # Significance level

    # FDR correction
    flat_pvals = pval_matrix.flatten()
    _, corrected_pvals = fdrcorrection(flat_pvals)
    pval_matrix = corrected_pvals.reshape(pval_matrix.shape)

    # Significance markers
    annotations = np.full_like(pval_matrix, "", dtype=object)
    for i in range(pval_matrix.shape[0]):
        for j in range(pval_matrix.shape[1]):
            p = pval_matrix[i, j]
            if p < 0.001:
                annotations[i, j] = "***"
            elif p < 0.01:
                annotations[i, j] = "**"
            elif p < 0.05:
                annotations[i, j] = "*"

    # Heatmap
    custom_cmap = LinearSegmentedColormap.from_list("custom", ["#61aacf", "#f9efef", "#da9599"])
    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data,
                xticklabels=list(AMINO_ACIDS),
                yticklabels=basic_labels,
                cmap=custom_cmap,
                annot=annotations,
                fmt='',
                cbar_kws={'label': 'Average Attribution Score'})
    plt.title("Average Attribution Scores on Train Dataset\n* = p < 0.05, ** = p < 0.01, *** = p < 0.001 (FDR corrected)")
    plt.xlabel("Amino Acid")
    plt.ylabel("Functional Tag")
    plt.tight_layout()
    path1 = os.path.join(save_dir, f"{prefix}_original_heatmap.png")
    plt.savefig(path1, dpi=300)
    plt.close()

    # Modify the clustered heatmap part - remove significance markers
    if cluster:
        # Convert data to DataFrame for clustering annotation
        attr_df = pd.DataFrame(
            heatmap_data,
            index=basic_labels,
            columns=list(AMINO_ACIDS)
        )

        # Plot clustered heatmap (without showing significance markers)
        g = sns.clustermap(
            attr_df,
            cmap=custom_cmap,
            row_cluster=True,
            col_cluster=True,
            cbar_kws={'label': 'Average Attribution Score'},
            figsize=(16, 8),
            tree_kws={'linewidths': 1.5}
        )

        # Adjust title and labels
        g.fig.suptitle("Clustered Attribution Scores on Train Dataset", y=1.05)
        g.ax_heatmap.set_xlabel("Amino Acid (Clustered)")
        g.ax_heatmap.set_ylabel("Functional Tag (Clustered)")

        # Save clustered heatmap
        path2 = os.path.join(save_dir, f"{prefix}_clustered_heatmap.png")
        plt.savefig(path2, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Clustered heatmap saved: {path2}")

    # Save CSV
    attr_df = pd.DataFrame(heatmap_data, index=basic_labels, columns=list(AMINO_ACIDS))
    pval_df = pd.DataFrame(pval_matrix, index=basic_labels, columns=list(AMINO_ACIDS))
    attr_df.to_csv(os.path.join(save_dir, f"{prefix}_attr_scores.csv"))
    pval_df.to_csv(os.path.join(save_dir, f"{prefix}_corrected_pvalues.csv"))

    print(f"Attribution scores CSV saved: {os.path.join(save_dir, f'{prefix}_attr_scores.csv')}")
    print(f"t-test values CSV saved: {os.path.join(save_dir, f'{prefix}_corrected_pvalues.csv')}")
    return heatmap_data, pval_matrix

# --------------------------
# 3. Main program workflow
# --------------------------
if __name__ == "__main__":
    # Read data
    csv_file_path = "./data/test_dataset.csv"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SequenceDataset(csv_file_path)
    data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_dynamic_padding, shuffle=False)

    tokenizer = EsmSequenceTokenizer()

    # Replace with the actual model
    model = load_model_from_checkpoint(PLiCat, f"best_model{epoch}.pt", device=device)
    model.to(device)
    model.eval()  # Switch to evaluation mode
    model.zero_grad()

    # Calculate attribution scores and generate heatmap
    attr_scores = compute_attributions(model, data_loader, tokenizer, device=device)
    heatmap_data, pval_matrix = compute_ttest_and_heatmap(attr_scores, save_dir="esmc-heatmaps",
                                                          prefix="train_dataset")

    del model  # Release model
    torch.cuda.empty_cache()  # Clear GPU cache