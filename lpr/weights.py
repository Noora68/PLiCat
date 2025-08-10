import torch
import json
from collections import defaultdict
import os
import numpy as np

def compute_class_weights(data_loader, category_mapping, save_path="./data", filename="class_weight.json"):
    """
    Compute class weights for multi-label classification to address class imbalance

    Parameters:
    data_loader -- DataLoader for training data (returns binary label vectors of length 9)
    category_mapping -- Dictionary mapping category indices to names
    save_path -- Directory path to save weights (optional)
    filename -- Filename for saving weights JSON (default: "class_weight.json")

    Returns:
    class_weights -- Tensor of computed class weights
    """
    # Initialize counters for positive class occurrences
    class_counts = defaultdict(int)
    total_samples = 0

    # Iterate through training batches to count class occurrences
    for batch in data_loader:
        # Convert labels to numpy for efficient processing
        labels_np = batch['labels'].numpy()
        total_samples += labels_np.shape[0]

        # Count positive samples for each class
        for class_idx in range(labels_np.shape[1]):
            class_counts[class_idx] += np.sum(labels_np[:, class_idx])

    # Initialize weights tensor
    num_classes = len(category_mapping)
    class_weights = torch.zeros(num_classes)

    # Calculate weights using inverse frequency scaling
    for class_idx in range(num_classes):
        count = class_counts.get(class_idx, 0)

        # Prevent division by zero
        if count == 0:
            count = 1e-6  # Small epsilon to avoid NaN

        # Weight formula: total_samples / (num_classes * positive_count)
        weight = total_samples / (num_classes * count)
        class_weights[class_idx] = weight

    # Save weights to JSON file if path is provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, filename)

        # Create serializable weight dictionary
        weights_dict = {
            "class_weights": class_weights.tolist(),
            "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            "total_samples": int(total_samples)
        }

        with open(full_path, "w") as f:
            json.dump(weights_dict, f, indent=4)
        print(f"\nMulti-label class weights saved to: {full_path}")

    return class_weights


def load_class_weights(weight_path, device="cpu"):
    """
    Load precomputed class weights from JSON file

    Parameters:
    weight_path -- Path to JSON file containing weight data
    device -- Target device for tensor (default: "cpu")

    Returns:
    class_weights -- Tensor of class weights loaded to specified device

    Raises:
    FileNotFoundError -- If weight file doesn't exist
    ValueError -- If file contains invalid JSON
    """
    try:
        with open(weight_path, "r") as f:
            weights_dict = json.load(f)

        # Convert list to tensor and move to device
        class_weights = torch.tensor(weights_dict["class_weights"])
        return class_weights.to(device)

    except FileNotFoundError:
        raise FileNotFoundError(f"Weight file {weight_path} not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {weight_path}")