import os
import sys
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Union, Tuple, Optional
# Find the project root directory and call os.path.dirname multiple times to find the directory upwards
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from PLiCat.utils import calculate_metrics, plot_roc_pr_curves,\
                  save_pred_results, process_labels, save_metrics_to_csv, \
                  plot_multilabel_confusion_matrix, load_model_from_checkpoint

from PLiCat.PLiCat_model import PLiCat
from PLiCat.PLiCat_dataset import SequenceDataset ,collate_fn_dynamic_padding

# Inference and evaluation
if __name__ == "__main__":
    batch_size = 32
    threshold = 0.6
    test_path = '../PLiCat/data/test_dataset.csv'  # Test dataset path
    
    # Create dataset and dataloader
    dataset = SequenceDataset(test_path)
    dataloader = DataLoader(dataset, batch_size=batch_size,  collate_fn=collate_fn_dynamic_padding, shuffle=False)  # shuffle must be False
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Evaluation scope: evaluate the specified model or the saved model for each round
    model_epochs = [e for e in range(6, 20)]
    models = [
       load_model_from_checkpoint(PLiCat, f"best_model{epoch}.pt", device=device)
       for epoch in model_epochs
    ]    # f"best_model{epoch}.pt" is replaced with the actual path

    if not models:
        raise ValueError("No models loaded. Please check model file paths.")

    # Initialize model index counter
    start = model_epochs[0]
    for i, model in enumerate(models):
        print(f"\n[Model {i+start}] Starting inference...")
        
        # Create output directory
        output_dir = f'results/result_{i+start}'
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize storage lists
        model_probs = []  # Current model's probability results
        true_labels = []
        sequences = []
        
        # Inference loop
        for batch in tqdm(dataloader, desc="Processing batches", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].float().to(device)

            with torch.no_grad():
                logits, _= model(input_ids, attention_mask)

            # Calculate and store probabilities
            probs = torch.sigmoid(logits)
            model_probs.append(probs.detach().cpu())  # Move to CPU and detach
            true_labels.append(labels.cpu())
            sequences.extend(batch['sequence'])
        
        # Concatenate batch results
        model_probs = torch.cat(model_probs, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
        
        # Apply threshold for binary predictions
        model_preds = (model_probs >= threshold).int()

        # Save results
        np.save(os.path.join(output_dir, "true_labels.npy"), true_labels.numpy())
        np.save(os.path.join(output_dir, "model_probs.npy"), model_probs.numpy())
        np.save(os.path.join(output_dir, "model_preds.npy"), model_preds.numpy())
        
        # Output shapes
        print("Probabilities shape:", model_probs.shape)
        print("Predictions shape:", model_preds.shape)
        print("\nProbabilities sample:")
        print(model_probs[:2])  # Print first 2 samples
        print("\nPredictions sample:")
        print(model_preds[:2])   # Print first 2 samples
    
        LABEL_NAMES = ['NO', 'FA', 'PR', 'GP', 'ST', 'PK', 'GL', 'SP', 'SL']

        # Save predictions to CSV
        save_pred_results(
            probs=model_probs.numpy(),
            preds=model_preds.numpy(),
            labels=true_labels.numpy(),
            sequences=sequences,
            label_names=LABEL_NAMES,
            output_path=os.path.join(output_dir, "predictions.csv")
        )
        print("All results saved to", output_dir)
        print(f"\n[Model {i+start}] Inference completed. Starting evaluation...")
    
        # ============ Evaluation =============== 
        # true_labels = np.load(os.path.join(output_dir, "true_labels.npy")).astype(int)
        # preds = np.load(os.path.join(output_dir, "model_preds.npy")).astype(int)
        # probs = np.load(os.path.join(output_dir, "model_probs.npy")).astype(np.float32)
        
        # Calculate evaluation metrics
        metrics = calculate_metrics(
            y_true=true_labels.numpy(),
            y_pred=model_preds.numpy(),
            y_prob=model_probs.numpy(),
            label_names=LABEL_NAMES
        )
        
        # Plot ROC and PR curves
        plot_roc_pr_curves(
            y_true=true_labels.numpy(),
            y_prob=model_probs.numpy(),
            label_names=LABEL_NAMES,
            output_dir=output_dir,
            axis_fontsize = 14,
            legend_fontsize = 12,
            line_width = 1.5 
        )
        
        # Plot confusion matrices
        plot_multilabel_confusion_matrix(
            y_true=true_labels.numpy(),
            y_pred=model_preds.numpy(),
            label_names=LABEL_NAMES,
            output_dir=output_dir
        )

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
        print(f"Macro ROC-AUC: {metrics['roc_auc_macro']:.4f}")
        print(f"Micro ROC-AUC: {metrics['roc_auc_micro']:.4f}")
        print(f"Macro PR-AUC: {metrics['pr_auc_macro']:.4f}")
        print(f"Micro PR-AUC: {metrics['pr_auc_micro']:.4f}")
        
        # Save metrics to CSV
        save_metrics_to_csv(metrics, os.path.join(output_dir, "evaluation_metrics.csv"))

        # Clean up resources
        del model
        torch.cuda.empty_cache()



