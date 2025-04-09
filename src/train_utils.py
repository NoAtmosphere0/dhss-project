#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training, evaluation, and utility functions for the
Multi-Task Learning model for ALASKA2 Image Steganalysis.
"""

from config import *


def train_epoch(
    model,
    dataloader,
    criterion_binary,
    criterion_multiclass,
    optimizer,
    device,
    multiclass_weight,
):
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model
        dataloader: Training data loader
        criterion_binary: Loss function for binary task
        criterion_multiclass: Loss function for multiclass task
        optimizer: PyTorch optimizer
        device: Device to use for training (CPU/GPU)
        multiclass_weight: Weight for multiclass loss

    Returns:
        tuple: (avg_loss, avg_binary_loss, avg_multiclass_loss)
    """
    model.train()
    total_loss = 0.0
    total_binary_loss = 0.0
    total_multiclass_loss = 0.0
    num_batches_multiclass_computed = 0  # To average multiclass loss correctly

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, binary_labels, multiclass_labels in progress_bar:
        images, binary_labels, multiclass_labels = (
            images.to(device),
            binary_labels.to(device),
            multiclass_labels.to(device),
        )

        optimizer.zero_grad()
        binary_logits, multiclass_logits = model(images)
        binary_logits = binary_logits.squeeze(1)

        # Binary Loss (all samples)
        loss_b = criterion_binary(binary_logits, binary_labels)

        # Multi-class Loss (only stego samples with valid labels)
        stego_indices = (binary_labels == 1).nonzero(as_tuple=True)[0]
        loss_m = torch.tensor(0.0, device=device)

        if stego_indices.numel() > 0:
            stego_multiclass_logits = multiclass_logits[stego_indices]
            stego_multiclass_labels = multiclass_labels[stego_indices]
            # Ensure we only compute loss for valid multiclass labels (e.g., ignore -1 if used)
            valid_stego_mask = stego_multiclass_labels >= 0
            if valid_stego_mask.sum() > 0:
                loss_m = criterion_multiclass(
                    stego_multiclass_logits[valid_stego_mask],
                    stego_multiclass_labels[valid_stego_mask],
                )
                num_batches_multiclass_computed += (
                    1  # Increment count if loss was computed
                )

        # Combine Losses
        combined_loss = loss_b + multiclass_weight * loss_m
        combined_loss.backward()
        optimizer.step()

        total_loss += combined_loss.item()
        total_binary_loss += loss_b.item()
        if loss_m.item() > 0:  # Only add if it was computed
            total_multiclass_loss += loss_m.item()

        progress_bar.set_postfix(
            {
                "Loss": f"{combined_loss.item():.4f}",
                "BinLoss": f"{loss_b.item():.4f}",
                "McLoss": f"{loss_m.item():.4f}" if loss_m.item() > 0 else "0.0",
            }
        )

    avg_loss = total_loss / len(dataloader)
    avg_binary_loss = total_binary_loss / len(dataloader)
    avg_multiclass_loss = (
        total_multiclass_loss / num_batches_multiclass_computed
        if num_batches_multiclass_computed > 0
        else 0.0
    )

    return avg_loss, avg_binary_loss, avg_multiclass_loss


def evaluate(
    model,
    dataloader,
    criterion_binary,
    criterion_multiclass,
    device,
    multiclass_weight=1.0,
):
    """
    Evaluate the model on validation data and calculate metrics.

    Args:
        model: PyTorch model
        dataloader: Validation data loader
        criterion_binary: Loss function for binary task
        criterion_multiclass: Loss function for multiclass task
        device: Device to use for evaluation (CPU/GPU)
        multiclass_weight: Weight for multiclass loss

    Returns:
        tuple: (avg_loss, avg_binary_loss, avg_multiclass_loss, metrics)
    """
    model.eval()
    total_loss = 0.0
    total_binary_loss = 0.0
    total_multiclass_loss = 0.0
    num_batches_multiclass_computed = 0

    all_binary_probs = []
    all_binary_labels = []
    all_multiclass_preds = []
    all_multiclass_labels = []
    all_is_stego_gt = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for images, binary_labels, multiclass_labels in progress_bar:
            images, binary_labels, multiclass_labels = (
                images.to(device),
                binary_labels.to(device),
                multiclass_labels.to(device),
            )

            binary_logits, multiclass_logits = model(images)
            binary_logits = binary_logits.squeeze(1)

            # Loss Calculation
            loss_b = criterion_binary(binary_logits, binary_labels)
            stego_indices = (binary_labels == 1).nonzero(as_tuple=True)[0]
            loss_m = torch.tensor(0.0, device=device)
            if stego_indices.numel() > 0:
                stego_multiclass_logits = multiclass_logits[stego_indices]
                stego_multiclass_labels = multiclass_labels[stego_indices]
                valid_stego_mask = stego_multiclass_labels >= 0
                if valid_stego_mask.sum() > 0:
                    loss_m = criterion_multiclass(
                        stego_multiclass_logits[valid_stego_mask],
                        stego_multiclass_labels[valid_stego_mask],
                    )
                    num_batches_multiclass_computed += 1

            combined_loss = loss_b + multiclass_weight * loss_m
            total_loss += combined_loss.item()
            total_binary_loss += loss_b.item()
            if loss_m.item() > 0:
                total_multiclass_loss += loss_m.item()

            # Store Predictions and Labels for Metrics
            binary_probs = torch.sigmoid(binary_logits)
            all_binary_probs.extend(binary_probs.cpu().numpy())  # Store probs for AUC
            all_binary_labels.extend(binary_labels.cpu().numpy().astype(int))
            all_is_stego_gt.extend(binary_labels.cpu().numpy().astype(int))

            multiclass_probs = torch.softmax(multiclass_logits, dim=1)
            multiclass_preds = torch.argmax(multiclass_probs, dim=1)
            all_multiclass_preds.extend(multiclass_preds.cpu().numpy())
            all_multiclass_labels.extend(multiclass_labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    avg_binary_loss = total_binary_loss / len(dataloader)
    avg_multiclass_loss = (
        total_multiclass_loss / num_batches_multiclass_computed
        if num_batches_multiclass_computed > 0
        else 0.0
    )

    # Calculate Metrics
    metrics = {}
    all_binary_labels = np.array(all_binary_labels)
    all_binary_probs = np.array(all_binary_probs)
    all_binary_preds = (all_binary_probs > 0.5).astype(int)  # Threshold probs

    metrics["binary_accuracy"] = accuracy_score(all_binary_labels, all_binary_preds)
    metrics["binary_precision"] = precision_score(
        all_binary_labels, all_binary_preds, zero_division=0
    )
    metrics["binary_recall"] = recall_score(
        all_binary_labels, all_binary_preds, zero_division=0
    )
    metrics["binary_f1"] = f1_score(
        all_binary_labels, all_binary_preds, zero_division=0
    )
    try:
        if len(np.unique(all_binary_labels)) > 1:
            metrics["binary_roc_auc"] = roc_auc_score(
                all_binary_labels, all_binary_probs
            )  # Use probs for AUC
        else:
            metrics["binary_roc_auc"] = float("nan")
    except ValueError as e:
        print(f"Could not calculate AUC: {e}")
        metrics["binary_roc_auc"] = float("nan")

    # Multi-class Metrics (ONLY on true stego samples with valid labels)
    all_multiclass_preds = np.array(all_multiclass_preds)
    all_multiclass_labels = np.array(all_multiclass_labels)
    all_is_stego_gt = np.array(all_is_stego_gt)

    stego_mask = (all_is_stego_gt == 1) & (all_multiclass_labels >= 0)
    stego_multiclass_preds = all_multiclass_preds[stego_mask]
    stego_multiclass_labels = all_multiclass_labels[stego_mask]

    if len(stego_multiclass_labels) > 0:
        metrics["multiclass_accuracy"] = accuracy_score(
            stego_multiclass_labels, stego_multiclass_preds
        )
        labels_range = list(range(NUM_CLASSES_MULTICLASS))  # Should be [0, 1, 2]
        # Ensure predicted labels don't exceed range (can happen with buggy models/data)
        stego_multiclass_preds_clipped = np.clip(
            stego_multiclass_preds, 0, NUM_CLASSES_MULTICLASS - 1
        )
        metrics["multiclass_confusion_matrix"] = confusion_matrix(
            stego_multiclass_labels, stego_multiclass_preds_clipped, labels=labels_range
        )
        cm = metrics["multiclass_confusion_matrix"]
        # Calculate per-class accuracy safely (avoid division by zero if a class has no samples in val set)
        class_counts = cm.sum(axis=1)
        per_class_acc = np.divide(
            cm.diagonal(),
            class_counts,
            out=np.zeros_like(cm.diagonal(), dtype=float),
            where=class_counts != 0,
        )
        metrics["multiclass_per_class_accuracy"] = per_class_acc.tolist()
    else:
        metrics["multiclass_accuracy"] = 0.0
        metrics["multiclass_confusion_matrix"] = np.zeros(
            (NUM_CLASSES_MULTICLASS, NUM_CLASSES_MULTICLASS), dtype=int
        )
        metrics["multiclass_per_class_accuracy"] = [0.0] * NUM_CLASSES_MULTICLASS
        print(
            "Warning: No valid stego samples found in validation set for multi-class evaluation."
        )

    return avg_loss, avg_binary_loss, avg_multiclass_loss, metrics


def predict_test(model, test_loader, device):
    """
    Generate predictions for the test set.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to use for prediction (CPU/GPU)

    Returns:
        DataFrame: Predictions for submission
    """
    model.eval()
    predictions = []
    image_ids = []

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Predicting Test", leave=False)
        for images, ids in progress_bar:
            images = images.to(device)
            binary_logits, _ = model(images)  # Only need binary output for submission
            binary_probs = torch.sigmoid(binary_logits.squeeze(1))

            predictions.extend(binary_probs.cpu().numpy())
            image_ids.extend(ids)

    # Create submission DataFrame
    submission_df = pd.DataFrame({"Id": image_ids, "Label": predictions})
    return submission_df


def log_metrics(epoch, train_metrics, val_metrics, log_file=None, backbone_type=None):
    """
    Log training and validation metrics to console and optionally to a file.

    Args:
        epoch: Current epoch number
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
        log_file: Path to log file (optional)
        backbone_type: Type of ResNet backbone (optional)
    """
    log_str = f"\n--- Epoch {epoch} Results ---\n"
    if backbone_type:
        log_str += f"Model: {backbone_type}\n"
    log_str += f"Training Losses:   Total={train_metrics['loss']:.4f}, Binary={train_metrics['binary_loss']:.4f}, Multiclass={train_metrics['multiclass_loss']:.4f}\n"
    log_str += f"Validation Losses: Total={val_metrics['loss']:.4f}, Binary={val_metrics['binary_loss']:.4f}, Multiclass={val_metrics['multiclass_loss']:.4f}\n"
    log_str += "Validation Metrics:\n"
    log_str += f"  Binary Accuracy:  {val_metrics['metrics']['binary_accuracy']:.4f}\n"
    log_str += f"  Binary Precision: {val_metrics['metrics']['binary_precision']:.4f}\n"
    log_str += f"  Binary Recall:    {val_metrics['metrics']['binary_recall']:.4f}\n"
    log_str += f"  Binary F1-Score:  {val_metrics['metrics']['binary_f1']:.4f}\n"
    log_str += f"  Binary ROC AUC:   {val_metrics['metrics']['binary_roc_auc']:.4f}\n"
    log_str += f"  Multiclass Accuracy (Stego Only): {val_metrics['metrics']['multiclass_accuracy']:.4f}\n"

    print(log_str)

    if log_file:
        with open(log_file, "a") as f:
            f.write(log_str)

    return log_str
