import re
import os
import sys
import json
import torch
from tqdm import tqdm
import pandas as pd
from captum.attr import LayerIntegratedGradients
from esm.tokenization import EsmSequenceTokenizer
from captum.attr import visualization as viz
from captum.attr._utils.visualization import format_word_importances
from torch.utils.data import Dataset, DataLoader
from IPython.display import display, HTML

# Find the project root directory and call os.path.dirname multiple times to find the directory upwards
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from lpr.LPRdataset import BindingDataset, binding_collate_fn_dynamic_padding
from lpr.LPRmodel import LPR


def tokens_to_str(tokens):
    # Filter special tokens and concatenate into a string
    special_tokens = {"<cls>", "<eos>", "<pad>", "<unk>"}  # Supplement according to your model's special tokens
    return "".join([tok for tok in tokens if tok not in special_tokens])


def construct_input_and_baseline(input_ids_list, tokenizer, pad_token_id, device):
    # Batch conversion
    token_list = [tokenizer.convert_ids_to_tokens(seq) for seq in input_ids_list]

    # Find the maximum sequence length
    max_len = max(len(seq) for seq in input_ids_list)

    # Pad sequences
    input_ids = [seq + [pad_token_id] * (max_len - len(seq)) for seq in input_ids_list]
    baseline_ids = [[0] + [pad_token_id] * (len(seq) - 2) + [2] for seq in input_ids_list]
    baseline_ids = [seq + [pad_token_id] * (max_len - len(seq)) for seq in baseline_ids]

    # Generate attention mask (1 for non-PAD tokens, 0 for PAD tokens)
    attention_mask = [[1 if token != pad_token_id else 0 for token in seq] for seq in input_ids]

    # Convert to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
    baseline_ids = torch.tensor(baseline_ids, dtype=torch.long).to(device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).to(device)

    return input_ids, baseline_ids, attention_mask, token_list


# Define forward_func: only output the logit you want to explain
def forward_func(input_ids, attention_mask=None):
    logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
    return logits  # shape: (batch, num_labels)


def visualize_text_custom(datarecords, label_dict, filename="attribution_visualization.html"):
    html = """
    <table border="0" cellpadding="5" style="border-collapse: collapse;">
    <thead>
    <tr>
        <th style="text-align: left;">True Label</th>
        <th style="text-align: left;">Predicted Label</th>
        <th style="text-align: left;">Source Sequence</th>
        <th style="text-align: left;">Attribution Score</th>
        <th style="text-align: left;"></th>
        <th style="text-align: left !important;">Amino Importance</th>
    </tr>
    </thead>
    """
    html += "<tbody>"
    for rec in datarecords:
        html += "<tr>"
        html += f'<td style="text-align: left;">{rec.true_class}</td>'

        match = re.search(r"tensor\((\d+),", str(rec.pred_class))
        if match:
            pred_class = int(match.group(1))
        else:
            pred_class = rec.pred_class
        cls_name = label_dict.get(str(pred_class), f"Unknown_{pred_class}")  # Map to name

        html += f'<td style="text-align: left;">{cls_name} ({rec.pred_prob:.2f})</td>'

        text = str(rec.attr_class)
        html += f'<td style="text-align: left;">{text[:5] + "..." if len(text) > 5 else text}</td>'

        html += f'<td style="text-align: left;">{rec.attr_score:.2f}</td>'
        html += f'<td style="text-align: left;">{format_word_importances(rec.raw_input_ids, rec.word_attributions)}</td>'
        html += "</tr>"

    html += "</tbody></table>"

    # Save as HTML file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Visualization results saved to {filename}")

    display(HTML(html))


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    norm = torch.norm(attributions)
    if norm != 0:
        attributions = attributions / norm
    return attributions


def interpret_sequence(input_ids_list, label_vec, true_class, tokenizer, device):
    input_ids, baseline_ids, attention_mask, token_list = \
        construct_input_and_baseline(input_ids_list, tokenizer, tokenizer.pad_token_id, device)
    special_tokens = {"<cls>", "<eos>", "<pad>", "<unk>"}  # Model special tokens

    # Batch processing
    clean_str_list = [tokens_to_str(tokens) for tokens in token_list]

    # No gradient retention during inference
    with torch.no_grad():
        # Model outputs logits
        pred_logits, _ = model(input_ids, attention_mask)

        # Convert to probabilities
        sigmoid_outputs = torch.sigmoid(pred_logits)
        t_predicted = (sigmoid_outputs > 0.6).int()

        # Extract indices from the first dimension (since pred_probs is (1, 9) for single sample multi-label)
        if t_predicted.any():
            # Get the first element of the tuple (class index) and squeeze to remove extra dimensions
            indices = t_predicted.nonzero(as_tuple=True)[1].squeeze()
            # Ensure indices is a list (avoid scalar when single element)
            if indices.dim() == 0:
                indices = indices.unsqueeze(0)  # Convert to [index]
        else:
            indices = torch.tensor([], dtype=torch.long)  # Empty indices

        # Use torch.index_select to ensure output is a tensor, compatible with scalar/list indices
        indices = indices.to(pred_logits[0].device)
        sigmoid_outputs = torch.index_select(sigmoid_outputs[0], dim=0, index=indices)

    records = []
    k = 0

    for tag_idx in range(len(label_vec)):
        if int(label_vec[tag_idx]) != 1:
            continue
        # Calculate attribution
        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=tag_idx,  # Index of the label to explain, e.g., the 0th label
            n_steps=50,
            return_convergence_delta=True
        )

        attributions_sum = summarize_attributions(attributions)
        if attributions_sum.dim() == 1:
            attributions_sum = attributions_sum.unsqueeze(0)  # Add dimension at dim 0 to become (1, seq_len)

        if k < len(indices):
            pred_prob = float(sigmoid_outputs[k])
            pred_class = int(indices[k])
            k += 1

            for i in range(len(token_list)):  # token_list_batch is a 2D list
                score_vis = viz.VisualizationDataRecord(
                    word_attributions=attributions_sum[i].tolist(),  # Attribution for this sentence

                    pred_prob=pred_prob,
                    pred_class=str(pred_class),

                    true_class=true_class,  # You can replace with actual label
                    attr_class=clean_str_list[i],
                    attr_score=torch.sum(attributions_sum[i]).item(),
                    raw_input_ids=token_list[i],  # Token list for this sentence
                    convergence_score=delta
                )
                records.append(score_vis)

    return records


def save_datarecords_to_csv(datarecords, filepath="./data/attribution_records.csv"):
    rows = []
    for rec in datarecords:
        # Extract integer from pred_class
        match = re.search(r"tensor\((\d+),", rec.pred_class)
        if match:
            pred_class = int(match.group(1))
        else:
            pred_class = rec.pred_class  # Store as-is if no match

        rows.append({
            "True Label": rec.true_class,
            "Predicted Label": pred_class,
            "Predicted Probability": round(rec.pred_prob, 4),
            "protein_Sequence": rec.attr_class,
            "Attribution Score": round(rec.attr_score, 4),
            "word_attributions": rec.word_attributions
        })

    # Save as CSV
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"CSV file saved successfully: {filepath}")


if __name__ == "__main__":

    tokenizer = EsmSequenceTokenizer()

    # Replace with actual data
    test_csv = './data/attribution_35_500_labeled.csv'

    dataset = BindingDataset(test_csv)
    data_loader = DataLoader(dataset, batch_size=1, collate_fn=binding_collate_fn_dynamic_padding, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Replace with actual model
    model = torch.load("model.pt", map_location=device, weights_only=False)  # Replace with actual model

    model.eval()  # Switch to evaluation mode
    model.zero_grad()  # Clear gradients (to avoid accumulation)

    # Select the layer to explain, such as the embedding layer
    layer = model.esmc.embed
    lig = LayerIntegratedGradients(forward_func, layer)

    # Read abbreviation.json
    # {
    #     "0": "No",
    #     "1": "FA",
    #     "2": "PR",
    #     "3": "GP",
    #     "4": "ST",
    #     "5": "PK",
    #     "6": "GL",
    #     "7": "SP",
    #     "8": "SL"
    # }
    with open("./data/abbreviation.json", "r", encoding="utf-8") as f:
        label_dict = json.load(f)

    all_records = []
    for batch in tqdm(data_loader, desc="Batch Attribution"):
        input_ids_list = batch['input_ids'].tolist()
        lipid_labels = batch['lipid_label']  # shape: (batch_size, 9), each row is 0/1 vector
        site_labels = batch['site_label']
        sequences = batch['sequence']

        # Convert labels to class name list
        for label_vec in lipid_labels:
            label_vec = label_vec.tolist()  # Convert to regular list
            class_names = ",".join([label_dict[str(i)] for i, val in enumerate(label_vec) if val == 1])
            records = interpret_sequence(input_ids_list, label_vec, class_names, tokenizer, device)
            all_records.extend(records)

    save_datarecords_to_csv(all_records, filepath="./data/binding_attribution_records.csv")

    # Custom format visualization
    visualize_text_custom(all_records, label_dict, filename="./data/binding_attribution_visualization.html")