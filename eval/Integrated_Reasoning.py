import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from tqdm import tqdm
import numpy as np

# Find the project root directory and call os.path.dirname multiple times to find the directory upwards
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from PLiCat.PLiCat_model import PLiCat
from PLiCat.PLiCat_dataset import SequenceDataset,collate_fn_dynamic_padding
from PLiCat.utils import calculate_metrics, plot_roc_pr_curves, plot_multilabel_confusion_matrix, \
    save_pred_results, save_metrics_to_csv, summarize_kfold_results, load_model_from_checkpoint

@torch.no_grad()
def ensemble_predict(
        models: List[torch.nn.Module],
        csv_file='',
        device: str = "cuda",
        method: str = "mean",
        weights: List[float] = None,
        threshold: float = 0.6,
        batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Perform ensemble prediction using multiple models.

    Args:
        models: List of model instances.
        csv_file: Path to the dataset for prediction.
        device: Computation device ('cuda' or 'cpu').
        method: Ensemble method ('mean' or 'weighted').
        weights: Weight list for weighted ensemble.
        threshold: Threshold for binarizing predictions.
        batch_size: Batch size for DataLoader.

    Returns:
        final_probs: Probability matrix [N, num_labels].
        final_preds: Binary prediction matrix [N, num_labels].
        true_labels: Ground truth label matrix [N, num_labels].
        all_sequences: List of sequences corresponding to predictions.
    """
    # 1. Prepare models
    for model in models:
        model.to(device)
        model.eval()  # Set model to evaluation mode

    # 2. Create dataset and dataloader
    dataset = SequenceDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_dynamic_padding,shuffle=False)

    # 3. Initialize storage lists
    all_probs = []  # Store probabilities from each model
    true_labels = []
    all_sequences = []

    # 4. Perform prediction for each model
    for i, model in enumerate(models):
        model_probs = []  # Probability results for the current model
        print(f'The {i+1}th model starts inference...')

        # Progress bar
        data_bar = tqdm(
            dataloader,
            desc=f"Inferencing...",
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )

        for batch in data_bar:
            # Move tensors to device and ensure float type
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].float().to(device)

            # Forward pass
            logits, _ = model(input_ids, attention_mask)

            # Calculate probabilities and store
            probs = torch.sigmoid(logits)
            model_probs.append(probs.detach().cpu())  # Move to CPU and detach from graph
            if i == 0:
                true_labels.append(labels.detach().cpu())
                all_sequences.extend(batch['sequence'])

        # Merge all batch results for the current model
        model_probs = torch.cat(model_probs, dim=0)
        all_probs.append(model_probs)

    if len(true_labels) > 0:
        true_labels = torch.cat(true_labels, dim=0)
    else:
        raise ValueError("true_labels is empty. Please check if dataloader is empty or model inference failed.")

    # 9. Stack results from all models [n_models, N, num_labels]
    stacked = torch.stack(all_probs)

    # 10. Apply ensemble method
    if method == "mean":
        final_probs = stacked.mean(dim=0)
    elif method == "weighted":
        if weights is None or len(weights) != len(models):
            raise ValueError("Number of weights must match number of models.")

        # Convert to tensor and normalize
        weight_tensor = torch.tensor(weights, dtype=stacked.dtype)
        weight_tensor = weight_tensor / weight_tensor.sum()

        # Apply weighted average
        final_probs = (stacked * weight_tensor.view(-1, 1, 1)).sum(dim=0)
    else:
        raise ValueError(f"Unknown ensemble method: {method}. Choose 'mean' or 'weighted'.")

    # 11. Apply threshold to get binary predictions
    final_preds = (final_probs >= threshold).int()

    return final_probs.numpy(), final_preds.numpy(), true_labels.numpy(), all_sequences


# Inference and evaluation
if __name__ == "__main__":

    # Create output directory
    output_dir = 'integration-results'
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If using weighted mode, provide metrics such as F1/AUC from each fold
    weights = [0.812, 0.801, 0.825, 0.798, 0.815, 0.811, 0.814, 0.799, 0.823, 0.820]

    test_path = '../PLiCat/data/test_dataset.csv'  # Test dataset path

    models = [
        load_model_from_checkpoint(PLiCat, "PLiCat-models/kfold/best_model_fold1.pt", device),
        load_model_from_checkpoint(PLiCat, "PLiCat-models/kfold/best_model_fold2.pt", device),
        load_model_from_checkpoint(PLiCat, "PLiCat-models/kfold/best_model_fold3.pt", device),
        load_model_from_checkpoint(PLiCat, "PLiCat-models/kfold/best_model_fold4.pt", device),
        load_model_from_checkpoint(PLiCat, "PLiCat-models/kfold/best_model_fold5.pt", device),
        load_model_from_checkpoint(PLiCat, "PLiCat-models/kfold/best_model_fold6.pt", device),
        load_model_from_checkpoint(PLiCat, "PLiCat-models/kfold/best_model_fold7.pt", device),
        load_model_from_checkpoint(PLiCat, "PLiCat-models/kfold/best_model_fold8.pt", device),
        load_model_from_checkpoint(PLiCat, "PLiCat-models/kfold/best_model_fold9.pt", device),
        load_model_from_checkpoint(PLiCat, "PLiCat-models/kfold/best_model_fold10.pt", device)
    ]

    # Perform ensemble prediction
    probs, preds, true_labels, all_sequences = ensemble_predict(
        models=models,
        csv_file=test_path,
        device=device,
        method="mean",
        weights=None,
        threshold=0.6,
        batch_size=32
    )

    # Save inference results
    np.save(os.path.join(output_dir, "ensemble_probs.npy"), probs)
    np.save(os.path.join(output_dir, "ensemble_preds.npy"), preds)
    np.save(os.path.join(output_dir, "true_labels.npy"), true_labels)

    LABEL_NAMES = ['NO', 'FA', 'PR', 'GP', 'ST', 'PK', 'GL', 'SP', 'SL']

    # Save model predictions to CSV
    save_pred_results(
        probs=probs,
        preds=preds,
        labels=true_labels,
        sequences=all_sequences,
        label_names=LABEL_NAMES,
        output_path=os.path.join(output_dir, "predictions.csv")
    )
    print("All results saved to", output_dir)

    # ============ Evaluation ===============
    # Read the .npy file if necessary
    # true_labels = np.load(os.path.join(output_dir, "true_labels.npy"))
    # preds = np.load(os.path.join(output_dir, "ensemble_preds.npy"))
    # probs = np.load(os.path.join(output_dir, "ensemble_probs.npy"))

    # Calculate evaluation metrics
    metrics = calculate_metrics(y_true=true_labels, y_pred=preds, y_prob=probs, label_names=LABEL_NAMES)

    # Plot ROC and PR curves
    plot_roc_pr_curves(y_true=true_labels, y_prob=probs, label_names=LABEL_NAMES, output_dir=output_dir, \
                       axis_fontsize = 14,   \
                       legend_fontsize = 12, \
                       line_width = 1.5 )

    # Plot multilabel confusion matrix
    plot_multilabel_confusion_matrix(y_true=true_labels, y_pred=preds, label_names=LABEL_NAMES, output_dir=output_dir)

    # Print key metrics
    print("\nEvaluation Metrics:")
    print(f"Sample Accuracy: {metrics['sample_accuracy']:.4f}")
    print(f"Label Accuracy: {metrics['label_accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Micro F1: {metrics['f1_micro']:.4f}")

    print(f"Macro Precision: {metrics['precision_macro']:.4f}")
    print(f"Micro Precision: {metrics['precision_micro']:.4f}")
    print(f"Macro Recall: {metrics['recall_macro']:.4f}")
    print(f"Micro Recall: {metrics['recall_micro']:.4f}")

    print(f"Macro roc-auc: {metrics['roc_auc_macro']:.4f}")
    print(f"Micro roc-auc: {metrics['roc_auc_micro']:.4f}")
    print(f"Macro pr-auc: {metrics['pr_auc_macro']:.4f}")
    print(f"Micro pr-auc: {metrics['pr_auc_micro']:.4f}")

    # Save metrics to CSV
    save_metrics_to_csv(metrics, os.path.join(output_dir, "evaluation_metrics.csv"))

