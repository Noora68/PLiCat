import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from utils import get_cosine_scheduler
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.utils.data import Subset, DataLoader
from esm.tokenization import EsmSequenceTokenizer
from LPRdataset import collate_fn_dynamic_padding

def train_model(model, train_loader, valid_loader, train_class_weights, category_mapping, fold,
                num_epochs=10, lr=2e-5, save_dir="lpr-models", early_stop_patience=5):
    """
    Train LPR model with progress tracking and validation monitoring

    Parameters:
    model -- LPR model to train
    data_loader -- Training data loader
    valid_loader -- Validation data loader
    train_class_weights -- Class weights tensor for training
    category_mapping -- Dictionary mapping category indices to names
    fold -- Current fold number (for cross-validation)
    num_epochs -- Number of training epochs (default 10)
    lr -- Learning rate (default 2e-5)
    save_dir -- Directory to save models and logs
    early_stop_patience -- Epochs to wait before early stopping (default 5)
    """
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    train_class_weights = train_class_weights.to(device)

    # Loss functions - Weighted for training, standard for validation
    train_criterion = nn.BCEWithLogitsLoss(weight=train_class_weights)
    val_criterion = nn.BCEWithLogitsLoss()

    # Optimizer configuration with parameter grouping
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.05},  # Apply weight decay to non-bias parameters
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}  # No decay for bias and LayerNorm parameters
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=lr)

    # Learning rate scheduler - Cosine annealing with warmup
    total_epochs = 30   #  Here, setting num_epochs to 30 is an empirical value
    steps_per_epoch = len(train_loader)
    scheduler = get_cosine_scheduler(optimizer, total_epochs, steps_per_epoch)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Still using the esm word segmentation method
    tokenizer = EsmSequenceTokenizer()

    # Training history tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_label_acc': [],  # Label-level accuracy
        'val_label_acc': [],  # Label-level accuracy
        'train_subset_acc': [],  # Sample-level accuracy
        'val_subset_acc': [],  # Sample-level accuracy
        'train_macro_f1': [],
        'train_weighted_f1': [],
        'val_macro_f1': [],
        'val_weighted_f1': [],
        'train_macro_precision': [],
        'val_macro_precision': [],
        'train_macro_recall': [],
        'val_macro_recall': [],
        'train_auc-roc': [],
        'val_auc-roc': [],
        'train_auc-pr': [],
        'val_auc-pr': []
    }

    # Training initialization
    print(f"\nStarting training for Fold {fold}...")
    best_val_loss = float('inf')
    patience_counter = 0
    log_file = os.path.join(save_dir, f"fold{fold}_training_log.txt")

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        all_train_labels, all_train_preds, all_train_probs = [], [], []

        # Training phase with progress bar
        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} [TRAIN]",
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )

        for batch in train_bar:
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].float().to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs, _ = model(input_ids, attention_mask)
            loss = train_criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()

            # Collect metrics
            batch_size = labels.size(0)
            train_loss += loss.item() * batch_size
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.6).float()

            all_train_labels.append(labels.cpu().detach().numpy())
            all_train_preds.append(preds.cpu().detach().numpy())
            all_train_probs.append(probs.cpu().detach().numpy())

            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{current_lr:.2e}"
            )

        # Calculate training metrics
        train_loss /= len(train_loader.dataset)
        all_train_labels = np.vstack(all_train_labels)
        all_train_preds = np.vstack(all_train_preds)
        all_train_probs = np.vstack(all_train_probs)

        train_label_acc = np.mean(all_train_labels == all_train_preds)  # Hamming accuracy
        train_subset_acc = accuracy_score(all_train_labels, all_train_preds)  # Exact match accuracy
        train_macro_f1 = f1_score(all_train_labels, all_train_preds, average='macro', zero_division=0)
        train_weighted_f1 = f1_score(all_train_labels, all_train_preds, average='weighted', zero_division=0)
        train_macro_precision = precision_score(all_train_labels, all_train_preds, average="macro", zero_division=0)
        train_macro_recall = recall_score(all_train_labels, all_train_preds, average="macro", zero_division=0)

        # AUC calculations with error handling
        try:
            train_auc_roc = roc_auc_score(all_train_labels, all_train_probs, average="macro")
            train_auc_pr = average_precision_score(all_train_labels, all_train_probs, average="macro")
        except Exception as e:
            print(f"[WARNING] Skipping training AUC calculations: {e}")
            train_auc_roc, train_auc_pr = 0.0, 0.0

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_labels, all_preds, all_probs = [], [], []

        val_bar = tqdm(
            valid_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} [VALID]",
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
            leave=False
        )

        with torch.no_grad():
            for batch in val_bar:

                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].float().to(device)

                outputs, _ = model(input_ids, attention_mask)
                loss = val_criterion(outputs, labels)

                batch_size = labels.size(0)
                val_loss += loss.item() * batch_size
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.6).float()

                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Calculate validation metrics
        val_loss /= len(valid_loader.dataset)
        all_labels = np.vstack(all_labels)
        all_preds = np.vstack(all_preds)
        all_probs = np.vstack(all_probs)

        val_label_acc = np.mean(all_labels == all_preds) # Hamming accuracy
        val_subset_acc = accuracy_score(all_labels, all_preds)  # Exact match accuracy
        val_macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        val_macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        val_macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

        try:
            val_auc_roc = roc_auc_score(all_labels, all_probs, average="macro")
            val_auc_pr = average_precision_score(all_labels, all_probs, average="macro")
            
        except Exception as e:
            print(f"[WARNING] Skipping validation AUC calculations: {e}")
            val_auc_roc, val_auc_pr = 0.0, 0.0

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_label_acc'].append(train_label_acc)
        history['val_label_acc'].append(val_label_acc)
        history['train_subset_acc'].append(train_subset_acc)
        history['val_subset_acc'].append(val_subset_acc)
        history['train_macro_f1'].append(train_macro_f1)
        history['train_weighted_f1'].append(train_weighted_f1)
        history['val_macro_f1'].append(val_macro_f1)
        history['val_weighted_f1'].append(val_weighted_f1)
        history['train_macro_precision'].append(train_macro_precision)
        history['val_macro_precision'].append(val_macro_precision)
        history['train_macro_recall'].append(train_macro_recall)
        history['val_macro_recall'].append(val_macro_recall)
        history['train_auc-roc'].append(train_auc_roc)
        history['train_auc-pr'].append(train_auc_pr)
        history['val_auc-roc'].append(val_auc_roc)
        history['val_auc-pr'].append(val_auc_pr)

        # Log epoch results
        epoch_time = time.time() - epoch_start
        epoch_summary = (
            f"\nEpoch [{epoch + 1}/{num_epochs}] completed in {epoch_time:.0f}s\n"
            f"Train Loss: {train_loss:.4f} | Train Label Acc: {train_label_acc:.4f} | Train Subset Acc: {train_subset_acc:.4f}\n"
            f"Valid Loss: {val_loss:.4f} | Valid Label Acc: {val_label_acc:.4f} | Valid Subset Acc: {val_subset_acc:.4f}\n"
            f"Train Macro F1: {train_macro_f1:.4f} | Valid Macro F1: {val_macro_f1:.4f}\n"
            f"Train auc-roc: {train_auc_roc:.4f} | Valid auc-roc: {val_auc_roc:.4f}\n"
            f"Train auc-pr: {train_auc_pr:.4f} | Valid auc-pr: {val_auc_pr:.4f}\n"
        )
        print(epoch_summary)

        # Save classification report
        with open(log_file, "a") as f:
            f.write(f"\n\n{'=' * 50}\nEpoch {epoch + 1} Results:\n{'=' * 50}")
            f.write(epoch_summary)
            f.write("\nClassification Report:\n")
            f.write(classification_report(
                all_labels,
                all_preds,
                target_names=list(category_mapping.values()),
                digits=4,
                zero_division=0
            ))
            f.write("\n" + "=" * 50 + "\n")

        # Early stopping and model checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(save_dir, f"best_model_fold{fold}.pt")

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'category_mapping': category_mapping,
                'model_config': getattr(model, 'config', None)
            }, best_model_path)
            print(f"Saved best model to {best_model_path} (Valid Loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {early_stop_patience} epochs without improvement")
                break

    print(f"\nTraining completed! Best Valid Loss: {best_val_loss:.4f}")

    # Save final artifacts
    history_path = os.path.join(save_dir, f"training_history_fold{fold}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")

    return history

def train_kfold_model(model_class, dataset, class_weights, category_mapping,
                      k=10, num_epochs=10, lr=2e-5, batch_size=16,
                      early_stop_patience=5, save_dir="lpr-models/kfold"):
    """
    Perform K-Fold cross-validation training

    Parameters:
    model_class -- Model class constructor
    dataset -- Complete dataset (torch.utils.data.Dataset)
    class_weights -- Function to compute class weights
    category_mapping -- Category index-to-name mapping
    k -- Number of folds (default 10)
    num_epochs -- Epochs per fold (default 10)
    lr -- Learning rate (default 2e-5)
    batch_size -- Batch size (default 16)
    early_stop_patience -- Early stopping patience (default 5)
    save_dir -- Base directory for saving results
    """

    fold_histories = []
    os.makedirs(save_dir, exist_ok=True)

    # Extract label matrix for stratified splitting
    all_labels = torch.stack([dataset[i]['labels'] for i in range(len(dataset))]).numpy()

    # Initialize multilabel stratified k-fold
    mskf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(mskf.split(np.zeros(len(dataset)), all_labels)):
        print(f"\n{'=' * 25} Fold {fold_idx + 1}/{k} {'=' * 25}")

        # Create data subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, collate_fn=collate_fn_dynamic_padding,
                                  shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, collate_fn=collate_fn_dynamic_padding,
                                shuffle=False)

        # Calculate fold-specific class weights
        fold_weights = class_weights(train_loader, category_mapping,
                                     save_path=save_dir,
                                     filename=f"fold{fold_idx + 1}_weights.json")

        # Initialize model
        model = model_class(num_classes=9)

        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=val_loader,
            train_class_weights=fold_weights,
            category_mapping=category_mapping,
            fold=fold_idx + 1,
            num_epochs=num_epochs,
            lr=lr,
            early_stop_patience=early_stop_patience,
            save_dir=save_dir
        )

        fold_histories.append(history)

    return fold_histories
