import pandas as pd
import ast
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib as mpl

# Set global font size
mpl.rcParams['font.size'] = 8

def plot_attribution_heatmap(sequence, attributions, binding_sites, protein_id, true_label, predict_label, predict_site,
                             metric, row_idx, outpath):
    # Calculate number of rows needed (50 residues per row)
    n = len(sequence)
    n_rows = (n + 49) // 50  # Round up

    # Create data matrix (fill empty cells with NaN)
    data = np.full((n_rows, 50), np.nan)
    char_matrix = [['' for _ in range(50)] for __ in range(n_rows)]

    # Skip the 0th position (CLS token) attribution values
    result = attributions[1:]

    # Fill data and character matrix
    for i, (char, attr) in enumerate(zip(sequence, result)):
        row_idx_pos = i // 50
        col_idx = i % 50
        data[row_idx_pos, col_idx] = attr
        char_matrix[row_idx_pos][col_idx] = char

    # Create figure
    fig, ax = plt.subplots(figsize=(15, max(3, n_rows * 0.7)))  # Ensure minimum height

    # Draw heatmap
    im = ax.imshow(data, cmap='coolwarm', aspect='auto', vmin=-10, vmax=10)

    # Add residue letters (below each heatmap cell)
    for i in range(n_rows):
        for j in range(50):
            if char_matrix[i][j]:  # Only process valid cells
                ax.text(j, i + 0.4, char_matrix[i][j],
                        ha='center', va='center', fontsize=6)

    # Mark binding sites (red border)
    for site in binding_sites:
        row = site // 50
        col = site % 50
        if row < n_rows and col < 50 and not np.isnan(data[row, col]):  # Ensure valid position
            rect = Rectangle((col - 0.5, row - 0.5), 1, 1,
                             linewidth=1.5, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    # Mark predicted sites (green border)
    for site in predict_site:
        row = site // 50
        col = site % 50
        if row < n_rows and col < 50 and not np.isnan(data[row, col]):  # Ensure valid position
            rect = Rectangle((col - 0.3, row - 0.3), 0.8, 0.8,
                             linewidth=1.5, edgecolor='green', facecolor='none')
            ax.add_patch(rect)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.5)
    cbar.set_label('Attribution Score × 20', rotation=270, labelpad=15, fontsize=10)

    # Set title and labels
    ax.set_title(f"Protein: {protein_id}  |  True: {true_label}  |  Predict: {predict_label} |  f1: {metric[2]} ",
                 fontsize=12)
    ax.set_xlabel('Residue Position', fontsize=12)
    ax.set_ylabel('Sequence Segment', fontsize=12)

    # Set axis ticks
    ax.set_xticks(np.arange(0, 50, 5))
    ax.set_xticklabels(np.arange(1, 51, 5), fontsize=10)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([f"Seg {i + 1}" for i in range(n_rows)], fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(outpath, f'protein_{protein_id}_row{row_idx + 1}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# RReplace with actual dataset
df = pd.read_csv('./data/attribution_records.csv')

# Process word_attributions column
def parse_attributions(attr_str):
    """Safely parse attribution score list"""
    try:
        # Try direct parsing
        return ast.literal_eval(attr_str)
    except:
        # Handle possible formatting issues
        cleaned = attr_str.replace('\n', '').replace(' ', '')
        if cleaned.startswith('[') and cleaned.endswith(']'):
            return ast.literal_eval(cleaned)
        else:
            # As a last resort, split the string
            return [float(x) for x in cleaned[1:-1].split(',') if x]

# Process site_label column
def parse_binding_sites(site_str):
    """Parse binding site positions"""
    sites = []
    for i, val in enumerate(site_str.split()):
        if val == '1':
            sites.append(i)
    return sites

# Process predict_site column
def parse_predict_sites(site_str):
    """Parse predicted site positions"""
    sites = []
    for i, val in enumerate(site_str.split(',')):
        numbers = re.findall(r'\d+', val)  # Match all continuous digits
        result = ''.join(numbers)  # Concatenate into full string
        sites.append(int(result) - 1)
    return sites

# Process metric column
def parse_metric_sites(metric_str):
    """Parse evaluation metric values"""
    metrics = []
    for i, val in enumerate(metric_str.split(',')):
        metrics.append(float(val))
    return metrics

basic_labels = ['NO', 'FA', 'PR', 'GP', 'ST', 'PK', 'GL', 'SP', 'SL']

outpath = './images'
os.makedirs(outpath, exist_ok=True)

# Process each row of data
for i, row in df.iterrows():
    sequence = row['protein_Sequence']

    # Parse attribution scores
    word_attributions = parse_attributions(row['word_attributions'])
    attributions = [float(x) * 20 for x in word_attributions]

    # Parse binding sites
    binding_sites = parse_binding_sites(row['site_label'])

    protein_id = row['protein_UniProt_ID']
    predict_site = parse_predict_sites(row['predict_site'])
    metric = parse_metric_sites(row['metric'])
    # Parse lipid label
    true_label = str(row['True Label'])
    index = int(str(row['Predicted Label']).strip())
    predict_label = basic_labels[index]

    # Plot heatmap
    plot_attribution_heatmap(sequence, attributions, binding_sites, protein_id, true_label, predict_label, predict_site,
                             metric, i, outpath)
