#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main training script for the Multi-Task Learning model for ALASKA2 Image Steganalysis.
Implements checkpoint saving and loading functionality for resuming training.
"""

from config import *
from dataset import create_data_loaders, prepare_data_splits
from model import MultiTaskResNet
from train_utils import evaluate, log_metrics, train_epoch


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Multi-Task Steganalysis Model")
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help="Number of epochs to train"
    )
    parser.add_argument(
        "--mc_weight",
        type=float,
        default=MULTICLASS_LOSS_WEIGHT,
        help="Weight for multi-class loss",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="ResNet backbone architecture",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--checkpoint_freq", type=int, default=1, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=BASE_DATA_DIR,
        help="Base directory for ALASKA2 dataset",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=VAL_SPLIT,
        help="Validation split ratio (0-1)",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    # Set random seed
    seed = set_seed(args.seed)
    print(f"Random seed set to {seed}")

    # Ensure checkpoint directory exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Setup device
    device = DEVICE
    print(f"Using device: {device}")

    # Map backbone string to numeric type
    backbone_map = {
        "resnet18": 18,
        "resnet34": 34,
        "resnet50": 50,
        "resnet101": 101,
        "resnet152": 152,
    }
    backbone_type = backbone_map.get(args.backbone, 18)

    # Create log directory and file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")

    # Create initial log entry
    with open(log_file, "w") as f:
        f.write(f"=== ALASKA2 Multi-Task Steganalysis Training ===\n")
        f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config:\n")
        for arg in vars(args):
            f.write(f"  {arg}: {getattr(args, arg)}\n")
        f.write(f"Device: {device}\n")
        f.write("\n")

    # Prepare data
    print("\n--- Preparing Data ---")
    try:
        train_paths, train_labels, val_paths, val_labels = prepare_data_splits(
            args.data_dir, val_split=args.val_split, random_seed=args.seed
        )
        train_loader, val_loader = create_data_loaders(
            train_paths, train_labels, val_paths, val_labels, batch_size=args.batch_size
        )
    except FileNotFoundError as e:
        print(f"Error preparing data: {e}")
        print(f"Please ensure the ALASKA2 dataset is available at {args.data_dir}")
        return

    # Initialize model, loss functions, optimizer
    print("\n--- Setting up Model ---")

    # Initialize training history
    train_history = {"loss": [], "binary_loss": [], "multiclass_loss": []}
    val_history = {
        "loss": [],
        "binary_loss": [],
        "multiclass_loss": [],
        "binary_accuracy": [],
        "binary_precision": [],
        "binary_recall": [],
        "binary_f1": [],
        "binary_roc_auc": [],
        "multiclass_accuracy": [],
    }

    # Variables to track training progress
    start_epoch = 0
    best_val_auc = 0.0
    best_model_path = None

    # Create model and optimizer
    model = MultiTaskResNet(
        num_classes_multiclass=NUM_CLASSES_MULTICLASS, 
        backbone_type=backbone_type,
        pretrained=True
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=2, verbose=True
    )

    # Loss functions
    criterion_binary = nn.BCEWithLogitsLoss()
    criterion_multiclass = nn.CrossEntropyLoss(ignore_index=-1)

    # Load checkpoint if specified
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"\n--- Loading Checkpoint: {args.checkpoint} ---")
            (
                model,
                optimizer,
                scheduler,
                start_epoch,
                best_val_auc,
                train_history,
                val_history,
            ) = MultiTaskResNet.load_checkpoint(
                args.checkpoint, device, optimizer, scheduler
            )
            print(f"Checkpoint loaded. Resuming from epoch {start_epoch+1}")
            print(f"Best validation AUC so far: {best_val_auc:.4f}")
        else:
            print(f"Warning: Checkpoint file not found at {args.checkpoint}")
            print("Training from scratch instead.")

    # Print model info
    print(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters"
    )

    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        current_epoch = epoch + 1
        print(f"\n--- Epoch {current_epoch}/{start_epoch + args.epochs} ---")

        # Train
        train_loss, train_bin_loss, train_mc_loss = train_epoch(
            model,
            train_loader,
            criterion_binary,
            criterion_multiclass,
            optimizer,
            device,
            args.mc_weight,
        )

        # Validate
        val_loss, val_bin_loss, val_mc_loss, val_metrics = evaluate(
            model,
            val_loader,
            criterion_binary,
            criterion_multiclass,
            device,
            args.mc_weight,
        )

        # Update learning rate scheduler
        current_auc = val_metrics.get("binary_roc_auc", 0.0)
        if not np.isnan(current_auc):
            scheduler.step(current_auc)
        else:
            scheduler.step(0.0)

        # Update histories
        train_history["loss"].append(train_loss)
        train_history["binary_loss"].append(train_bin_loss)
        train_history["multiclass_loss"].append(train_mc_loss)

        val_history["loss"].append(val_loss)
        val_history["binary_loss"].append(val_bin_loss)
        val_history["multiclass_loss"].append(val_mc_loss)
        val_history["binary_accuracy"].append(val_metrics["binary_accuracy"])
        val_history["binary_precision"].append(val_metrics["binary_precision"])
        val_history["binary_recall"].append(val_metrics["binary_recall"])
        val_history["binary_f1"].append(val_metrics["binary_f1"])
        val_history["binary_roc_auc"].append(val_metrics["binary_roc_auc"])
        val_history["multiclass_accuracy"].append(val_metrics["multiclass_accuracy"])

        # Log metrics
        train_metrics = {
            "loss": train_loss,
            "binary_loss": train_bin_loss,
            "multiclass_loss": train_mc_loss,
        }
        val_metrics_dict = {
            "loss": val_loss,
            "binary_loss": val_bin_loss,
            "multiclass_loss": val_mc_loss,
            "metrics": val_metrics,
        }
        log_metrics(current_epoch, train_metrics, val_metrics_dict, log_file)

        # Save regular checkpoint based on frequency
        if current_epoch % args.checkpoint_freq == 0:
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f"checkpoint_epoch_{current_epoch}.pth"
            )
            model.save_checkpoint(
                checkpoint_path,
                optimizer,
                scheduler,
                current_epoch,
                best_val_auc,
                train_history,
                val_history,
            )
            print(f"Checkpoint saved to {checkpoint_path}")

        # Save best model based on validation AUC
        if not np.isnan(current_auc) and current_auc > best_val_auc:
            best_val_auc = current_auc

            # Remove old best model if it exists
            if best_model_path and os.path.exists(best_model_path):
                try:
                    os.remove(best_model_path)
                except OSError:
                    pass

            # Save new best model
            best_model_path = os.path.join(
                CHECKPOINT_DIR, f"best_model_auc_{best_val_auc:.4f}.pth"
            )
            model.save_checkpoint(
                best_model_path,
                optimizer,
                scheduler,
                current_epoch,
                best_val_auc,
                train_history,
                val_history,
            )
            print(
                f"  ----> Best Model Saved to {best_model_path} (Val AUC: {best_val_auc:.4f}) <----"
            )

    print("\n--- Training Complete ---")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training logs saved to: {log_file}")

    # Save final model
    final_model_path = os.path.join(CHECKPOINT_DIR, f"final_model.pth")
    model.save_checkpoint(
        final_model_path,
        optimizer,
        scheduler,
        start_epoch + args.epochs,
        best_val_auc,
        train_history,
        val_history,
    )
    print(f"Final model saved to: {final_model_path}")

    # Save training history as JSON
    history_path = os.path.join(log_dir, "training_history.json")
    with open(history_path, "w") as f:
        history = {
            "train": train_history,
            "val": val_history,
            "best_val_auc": best_val_auc,
            "total_epochs": start_epoch + args.epochs,
        }
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
