#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prediction script for the Multi-Task Learning model for ALASKA2 Image Steganalysis.
Loads a trained model checkpoint and generates predictions on test data.
"""

from config import *
from dataset import create_test_loader
from model import MultiTaskResNet
from train_utils import predict_test


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate Predictions with Trained Model"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=TEST_DIR,
        help="Directory containing test images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Output file path for predictions",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE * 2,
        help="Batch size for prediction",
    )
    return parser.parse_args()


def main():
    """Main prediction function"""
    args = parse_args()

    # Set up device
    device = DEVICE
    print(f"Using device: {device}")

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    # Load model from checkpoint
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model, _, _, epoch, val_auc, _, _ = MultiTaskResNet.load_checkpoint(
        args.checkpoint, device
    )
    print(
        f"Model loaded successfully (from epoch {epoch}, validation AUC: {val_auc:.4f})"
    )

    # Create test data loader
    print(f"Loading test data from: {args.test_dir}")
    try:
        test_loader = create_test_loader(args.test_dir, batch_size=args.batch_size)
    except FileNotFoundError as e:
        print(f"Error loading test data: {e}")
        return

    # Generate predictions
    print("Generating predictions...")
    submission_df = predict_test(model, test_loader, device)

    # Save predictions
    print(f"Saving predictions to: {args.output}")
    submission_df.to_csv(args.output, index=False)
    print(f"Done! Saved {len(submission_df)} predictions.")


if __name__ == "__main__":
    main()
