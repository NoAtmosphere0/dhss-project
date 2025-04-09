#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive demo for ALASKA2 Multi-Task Steganalysis.
Loads a random image, displays it, and runs inference with a selected checkpoint.
"""

import os
import sys
import random
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

from config import *  # This already includes val_transform
from model import MultiTaskResNet
from dataset import Alaska2Dataset


def get_available_checkpoints():
    """Get list of available checkpoint files"""
    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Error: Checkpoint directory '{CHECKPOINT_DIR}' not found")
        return []
    
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth"))
    checkpoints.append("multitask_steganalysis_resnet18_best_auc_0.5029.pth")
    
    # Sort by modification time (most recent first)
    checkpoints.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
    return checkpoints


def get_random_images(base_dir, num_images=1):
    """Get random images from Cover and Stego directories for demo"""
    image_paths = []
    labels = []
    ground_truth = []
    
    # Categories and their corresponding directories
    categories = {
        "Cover": COVER_DIR,
        "JMiPOD": JMIPOD_DIR,
        "JUNIWARD": JUNIWARD_DIR,
        "UERD": UERD_DIR,
    }
    
    for category, directory in categories.items():
        if os.path.exists(directory):
            files = glob.glob(os.path.join(directory, "*.jpg"))
            if files:
                for _ in range(num_images // len(categories)):
                    img_path = random.choice(files)
                    image_paths.append(img_path)
                    
                    # Binary label (0=Cover, 1=Stego)
                    binary_label = 0 if category == "Cover" else 1
                    labels.append(binary_label)
                    
                    # Full classification info
                    ground_truth.append({
                        "path": img_path,
                        "category": category,
                        "binary_label": binary_label,
                        "multiclass_label": -1 if category == "Cover" else 
                                          0 if category == "JMiPOD" else
                                          1 if category == "JUNIWARD" else 2
                    })
    
    return image_paths, labels, ground_truth


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    try:
        print(f"Loading checkpoint: {checkpoint_path}")
        model, _, _, _, _, _, _ = MultiTaskResNet.load_checkpoint(
            checkpoint_path, device
        )
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def process_image(image_path):
    """Load and preprocess an image"""
    image = Image.open(image_path).convert("RGB")
    processed_image = val_transform(image)
    return image, processed_image.unsqueeze(0)  # Add batch dimension


def run_inference(model, image_tensor, device):
    """Run inference on an image"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        binary_logits, multiclass_logits = model(image_tensor)
        
        # Get binary prediction (0=Cover, 1=Stego)
        binary_prob = torch.sigmoid(binary_logits).item()
        binary_pred = 1 if binary_prob > 0.5 else 0
        
        # Get multiclass prediction (only relevant if binary_pred is 1)
        if binary_pred == 1:
            multiclass_prob = torch.softmax(multiclass_logits, dim=1)
            multiclass_pred = torch.argmax(multiclass_prob, dim=1).item()
            multiclass_probs = multiclass_prob.squeeze().cpu().numpy()
        else:
            multiclass_pred = -1
            multiclass_probs = None
            
        return {
            "binary_pred": binary_pred,
            "binary_prob": binary_prob,
            "multiclass_pred": multiclass_pred,
            "multiclass_probs": multiclass_probs
        }


def display_results(image, results, ground_truth):
    """Display the image and prediction results"""
    # Setup figure
    plt.figure(figsize=(10, 8))
    
    # Display image
    plt.subplot(1, 1, 1)
    plt.imshow(image)
    plt.axis('off')
    
    # Generate title with predictions
    title = f"File: {os.path.basename(ground_truth['path'])}\n"
    title += f"Ground Truth: {ground_truth['category']}\n"
    
    # Binary prediction results
    binary_result = "Stego" if results["binary_pred"] == 1 else "Cover"
    binary_correct = results["binary_pred"] == ground_truth["binary_label"]
    title += f"Prediction: {binary_result} (Confidence: {results['binary_prob']:.4f})\n"
    
    # Add a correctly/incorrectly tag
    if binary_correct:
        title += "Binary Classification: \u2713 CORRECT"
    else:
        title += "Binary Classification: \u2717 INCORRECT"
    
    # Add multiclass info if it's a stego image
    if results["binary_pred"] == 1 and results["multiclass_probs"] is not None:
        stego_types = ["JMiPOD", "JUNIWARD", "UERD"]
        detected_type = stego_types[results["multiclass_pred"]]
        
        title += f"\nDetected Stego Type: {detected_type}"
        
        # If ground truth is also stego, show if multiclass prediction was correct
        if ground_truth["binary_label"] == 1:
            true_type = stego_types[ground_truth["multiclass_label"]]
            multiclass_correct = results["multiclass_pred"] == ground_truth["multiclass_label"]
            
            if multiclass_correct:
                title += f" (\u2713 CORRECT)"
            else:
                title += f" (\u2717 INCORRECT, should be {true_type})"
                
        # Show all class probabilities
        title += "\nStego Type Probabilities:"
        for i, stego_type in enumerate(stego_types):
            title += f"\n - {stego_type}: {results['multiclass_probs'][i]:.4f}"
    
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Interactive Steganalysis Demo")
    parser.add_argument("--checkpoint", help="Checkpoint file to use (will show selection menu if not specified)")
    parser.add_argument("--num_images", type=int, default=1, help="Number of random images to process")
    args = parser.parse_args()
    
    # Get device
    device = DEVICE
    print(f"Using device: {device}")
    
    # Get list of checkpoints
    available_checkpoints = get_available_checkpoints()
    if not available_checkpoints:
        print("No checkpoints found. Please train a model first.")
        sys.exit(1)
    
    # Select checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        print("\nAvailable checkpoints:")
        for i, cp in enumerate(available_checkpoints):
            print(f"{i+1}. {os.path.basename(cp)}")
        
        try:
            selection = int(input("\nSelect checkpoint (number): ")) - 1
            checkpoint_path = available_checkpoints[selection]
        except (ValueError, IndexError):
            print("Invalid selection. Using the most recent checkpoint.")
            checkpoint_path = available_checkpoints[0]
    
    # Load model
    model = load_model(checkpoint_path, device)
    if model is None:
        print("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Get random images
    image_paths, labels, ground_truth = get_random_images(BASE_DATA_DIR, args.num_images)
    
    if not image_paths:
        print("No images found. Check that your data directories are correctly set.")
        sys.exit(1)
    
    # Process each image
    for idx, (img_path, gt) in enumerate(zip(image_paths, ground_truth)):
        print(f"\nProcessing image {idx+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        
        # Load and process image
        orig_image, processed_image = process_image(img_path)
        
        # Run inference
        results = run_inference(model, processed_image, device)
        
        # Display results
        print(f"Ground Truth: {gt['category']}")
        print(f"Prediction: {'Stego' if results['binary_pred'] == 1 else 'Cover'} " +
              f"(Confidence: {results['binary_prob']:.4f})")
        
        # Display the image with results
        display_results(orig_image, results, gt)


if __name__ == "__main__":
    main()