import os
import json
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    multilabel_confusion_matrix
)


def get_cosine_scheduler(optimizer, total_epochs, steps_per_epoch):
    """
    Create a learning rate scheduler with cosine annealing and warmup

    Parameters:
    optimizer -- Optimizer to attach scheduler to
    total_epochs -- Total number of training epochs
    steps_per_epoch -- Number of optimization steps per epoch

    Returns:
    LambdaLR scheduler implementing cosine decay with linear warmup
    """
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = max(1, int(0.1 * total_steps))  # First 10% for warmup
    min_lr_ratio = 0.1  # Minimum LR is 10% of max LR

    def lr_lambda(step):
        # Linear warmup phase
        if step < warmup_steps:
            return step / warmup_steps
        # Cosine annealing phase
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def summarize_kfold_results(save_dir="models/kfold", k_folds=10):
    """
    Aggregate best results from K-fold cross-validation

    Parameters:
    save_dir -- Directory containing fold history files
    k_folds -- Number of folds used in cross-validation

    Returns:
    Dictionary of average metrics across all folds
    """
    # Define metrics to aggregate
    metrics = [
        "train_loss", "val_loss",
        "train_label_acc", "val_label_acc",
        "train_subset_acc", "val_subset_acc",
        "train_macro_f1", "train_weighted_f1",
        "val_macro_f1", "val_weighted_f1",
        "train_auc-roc", "train_auc-pr", "train_auc-pr-label",
        "val_auc-roc", "val_auc-pr", "val_auc-pr-label"
    ]

    # Initialize results container
    avg_results = {f"avg_{m}": 0.0 for m in metrics}

    for fold in range(1, k_folds + 1):
        path = os.path.join(save_dir, f"training_history_fold{fold}.json")
        if not os.path.exists(path):
            print(f"[WARNING] History file not found for fold {fold}: {path}")
            continue

        with open(path, 'r', encoding='utf-8') as f:
            history = json.load(f)

        # Find best epoch (minimum validation loss)
        val_loss_list = history.get("val_loss", [])
        if not val_loss_list:
            print(f"[WARNING] Missing val_loss for fold {fold}, skipping")
            continue

        best_epoch = int(np.argmin(val_loss_list))

        # Aggregate metrics from best epoch
        for m in metrics:
            metric_list = history.get(m, [])
            if best_epoch < len(metric_list):
                avg_results[f"avg_{m}"] += metric_list[best_epoch] / k_folds
            else:
                print(f"[WARNING] Metric {m} missing for fold {fold}, skipping")

    # Print and save results
    print(f"\n{k_folds}-fold average results (best epoch per fold):")
    for k, v in avg_results.items():
        print(f"{k}: {v:.4f}")

    output_path = os.path.join(save_dir, f"kfolds1-{k_folds}_avg_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(avg_results, f, indent=4, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")

    return avg_results


def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        label_names: List[str]
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for multi-label classification

    Parameters:
    y_true -- Ground truth labels (n_samples, n_classes)
    y_pred -- Predicted labels (n_samples, n_classes)
    y_prob -- Predicted probabilities (n_samples, n_classes)
    label_names -- Names of each class

    Returns:
    Dictionary containing all computed metrics
    """
    metrics = {}

    # Sample-level metrics
    metrics["sample_accuracy"] = np.mean(np.all(y_true == y_pred, axis=1))  # Exact match

    # Label-level metrics
    metrics["label_accuracy"] = np.mean(y_true == y_pred)  # Hamming accuracy

    # Macro-averaged metrics
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Micro-averaged metrics
    metrics["precision_micro"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["recall_micro"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["f1_micro"] = f1_score(y_true, y_pred, average="micro", zero_division=0)

    # AUC metrics
    try:
        metrics["roc_auc_macro"] = roc_auc_score(y_true, y_prob, average="macro")
        metrics["roc_auc_micro"] = roc_auc_score(y_true, y_prob, average="micro")
    except ValueError:
        pass  # Handle cases with single-class labels

    # PR AUC metrics
    metrics["pr_auc_macro"] = average_precision_score(y_true, y_prob, average="macro")
    metrics["pr_auc_micro"] = average_precision_score(y_true, y_prob, average="micro")

    # Label-level AUC-PR
    ap_per_label = average_precision_score(y_true, y_prob, average=None)
    metrics["pr_auc_label"] = ap_per_label.mean()

    # Per-class metrics
    for i, name in enumerate(label_names):
        metrics[f"{name}_precision"] = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
        metrics[f"{name}_recall"] = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        metrics[f"{name}_f1"] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        metrics[f"{name}_accuracy"] = accuracy_score(y_true[:, i], y_pred[:, i])
        try:
            metrics[f"{name}_roc_auc"] = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            metrics[f"{name}_roc_auc"] = float('nan')  # Handle single-class cases
        metrics[f"{name}_pr_auc"] = average_precision_score(y_true[:, i], y_prob[:, i])

    return metrics


def plot_roc_pr_curves(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        label_names: List[str],
        output_dir: str,
        axis_fontsize: int = 14,
        legend_fontsize: int = 12,
        line_width: float = 1.5
):
    os.makedirs(output_dir, exist_ok=True)

    # ROC curves
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(label_names):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = roc_auc_score(y_true[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, lw=line_width, label=f'{name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=line_width)  # Random classifier line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=axis_fontsize)
    plt.ylabel('True Positive Rate', fontsize=axis_fontsize)
    plt.title('ROC Curves per Label', fontsize=axis_fontsize)
    plt.tick_params(axis='both', labelsize=axis_fontsize)
    plt.legend(loc="lower right", fontsize=legend_fontsize)
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300)
    plt.close()

    # Precision-Recall curves
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(label_names):
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        pr_auc = average_precision_score(y_true[:, i], y_prob[:, i])
        plt.plot(recall, precision, lw=line_width, label=f'{name} (AUC = {pr_auc:.2f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=axis_fontsize)
    plt.ylabel('Precision', fontsize=axis_fontsize)
    plt.title('Precision-Recall Curves per Label', fontsize=axis_fontsize)
    plt.tick_params(axis='both', labelsize=axis_fontsize)
    plt.legend(loc="lower left", fontsize=legend_fontsize)
    plt.savefig(os.path.join(output_dir, 'pr_curves.png'), dpi=300)
    plt.close()

def plot_multilabel_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_names: List[str],
        output_dir: str,
        figsize=(10, 8)
):
    """
    Generate a confusion matrix for multi-label classification that resembles
    a standard confusion matrix.

    Parameters:
    y_true -- Ground truth labels (n_samples, n_classes)
    y_pred -- Predicted labels (n_samples, n_classes)
    label_names -- Names of each class
    output_dir -- Directory to save plot
    figsize -- Figure dimensions (default (10,8))

    Matrix interpretation:
    - Rows: True labels (with total counts shown in labels)
    - Diagonal: Correct predictions (true A and predicted A)
    - Off-diagonal: Misclassifications (true A but predicted B)
    """
    os.makedirs(output_dir, exist_ok=True)
    num_classes = len(label_names)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    # Precompute true label counts for each class
    true_label_counts = y_true.sum(axis=0)
    pred_label_counts = y_pred.sum(axis=0)
    
    # Build confusion matrix
    for i in range(len(y_true)):
        true_labels = np.where(y_true[i] == 1)[0]
        pred_labels = np.where(y_pred[i] == 1)[0]
        
        for t in true_labels:
            if t in pred_labels:  # Correct prediction: true A predicted A
                matrix[t, t] += 1
            else:  # Misclassification: true A predicted as other labels
                for p in pred_labels:
                    matrix[t, p] += 1
                    
    # Create y-tick labels with true counts
  
    yticklabels = [
        f"{name} ({int(true_label_counts[i])}/{int(true_label_counts[i]-matrix[i, :].sum())})" 
        for i, name in enumerate(label_names)
    ]
    
    # Create heatmap visualization
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=label_names, yticklabels=yticklabels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label (with sample counts)')
    plt.title('Multi-label Confusion Matrix')
    plt.tight_layout()

    save_path = os.path.join(output_dir, 'multilabel_confusion_matrix.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")




def save_pred_results(
        probs: np.ndarray,
        preds: np.ndarray,
        labels: Optional[np.ndarray],
        sequences: Optional[List[str]],
        label_names: List[str],
        output_path: str
):
    """
    Save prediction results to CSV file

    Parameters:
    probs -- Predicted probabilities (n_samples, n_classes)
    preds -- Predicted labels (n_samples, n_classes)
    labels -- Ground truth labels (n_samples, n_classes) (optional)
    sequences -- Input sequences (n_samples) (optional)
    label_names -- Names of each class
    output_path -- File path to save CSV
    """
    results = []

    # Build result dictionary for each sample
    for i in range(len(probs)):
        sample_data = {"sequence": sequences[i] if sequences else f"sample_{i}"}

        if labels is not None:
            for j, name in enumerate(label_names):
                sample_data[f"true_{name}"] = labels[i, j]

        for j, name in enumerate(label_names):
            sample_data[f"prob_{name}"] = probs[i, j]
            sample_data[f"pred_{name}"] = preds[i, j]

        results.append(sample_data)

    # Save as CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Prediction results saved to {output_path}")


def process_labels(csv_path) -> tuple:
    """
    Process CSV file with sequence-label data

    Parameters:
    csv_path -- Path to CSV file with 'Sequence' and 'Label' columns

    Returns:
    Tuple containing:
        sequences: List of sequence strings
        labels: NumPy array of label vectors (n_samples, n_classes)

    Raises:
    ValueError if required columns are missing
    """
    df = pd.read_csv(csv_path)

    # Validate required columns
    if 'Sequence' not in df.columns or 'Label' not in df.columns:
        raise ValueError("CSV must contain 'Sequence' and 'Label' columns")

    # Convert label strings to vectors
    label_vectors = []
    for label_str in df['Label']:
        vector = np.array([int(x) for x in label_str.strip().split()])
        label_vectors.append(vector)

    return df['Sequence'].tolist(), np.array(label_vectors)


def save_metrics_to_csv(metrics: dict, output_path: str) -> None:
    """
    Save evaluation metrics dictionary to CSV

    Parameters:
    metrics -- Dictionary of metric names to values
    output_path -- File path to save CSV (including filename)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to {output_path}")


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """
    Calculate binary Shannon entropy for probability vectors

    Parameters:
    p -- Probability array (n_samples, n_classes)

    Returns:
    Entropy values for each sample (n_samples)
    """
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return - (p * np.log2(p) + (1 - p) * np.log2(1 - p)).sum(axis=1)


def shannon_entropy_batch(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculate Shannon entropy for batch of logits

    Parameters:
    logits -- Model outputs before sigmoid (batch_size, n_classes)

    Returns:
    Entropy values for each sample in batch (batch_size)
    """
    probs = torch.sigmoid(logits)
    eps = 1e-12
    probs = torch.clamp(probs, eps, 1 - eps)
    entropy = - (probs * torch.log2(probs) + (1 - probs) * torch.log2(1 - probs))
    return entropy.sum(dim=1)  # Sum across classes


def shannon_entropy_per_label(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculate average entropy per class label

    Parameters:
    logits -- Model outputs before sigmoid (batch_size, n_classes)

    Returns:
    Average entropy per class (n_classes)
    """
    probs = torch.sigmoid(logits)
    eps = 1e-12
    probs = torch.clamp(probs, eps, 1 - eps)
    entropy = - (probs * torch.log2(probs) + (1 - probs) * torch.log2(1 - probs))
    return entropy.mean(dim=0)  # Average across samples


def load_model_from_checkpoint(model_class, checkpoint_path, device="cuda"):
    """
    Load model from training checkpoint

    Parameters:
    model_class -- Model class constructor
    checkpoint_path -- Path to model checkpoint file
    device -- Target device ('cuda' or 'cpu')

    Returns:
    Loaded model in evaluation mode
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model_class()  # Initialize model architecture
    model.load_state_dict(checkpoint['model_state_dict'])  # Load weights
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model

