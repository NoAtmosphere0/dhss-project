#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive Streamlit demo for ALASKA2 Multi-Task Steganalysis.
Loads images, displays them, and runs inference with selected checkpoints.
"""

import os
import random
import glob
import numpy as np
import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

from config import *  # This includes val_transform
from model import MultiTaskResNet

# Set page configuration
st.set_page_config(
    page_title="Steganalysis Demo",
    page_icon="🔍",
    layout="wide",
)

# Custom color scheme for confidence visualization
colors = [(0.9, 0.17, 0.31), (0.9, 0.6, 0.18), (0.13, 0.65, 0.4)]
custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=100)


def get_available_checkpoints():
    """Get list of available checkpoint files"""
    if not os.path.exists(CHECKPOINT_DIR):
        st.error(f"Error: Checkpoint directory '{CHECKPOINT_DIR}' not found")
        return []
    
    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth"))
    
    # Add default checkpoint if it exists
    default_checkpoint = "multitask_steganalysis_resnet18_best_auc_0.5029.pth"
    if os.path.exists(default_checkpoint):
        checkpoints.append(default_checkpoint)
    
    # Sort by modification time (most recent first)
    checkpoints.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True)
    return checkpoints


@st.cache_resource
def load_model(checkpoint_path, device):
    """Load model from checkpoint (with caching)"""
    try:
        model, _, _, _, _, _, _ = MultiTaskResNet.load_checkpoint(
            checkpoint_path, device
        )
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def get_random_images(base_dir, num_images=1, category_filter=None):
    """Get random images from specified categories"""
    image_paths = []
    ground_truth = []
    
    # Categories and their corresponding directories
    categories = {
        "Cover": COVER_DIR,
        "JMiPOD": JMIPOD_DIR,
        "JUNIWARD": JUNIWARD_DIR,
        "UERD": UERD_DIR,
    }
    
    # Filter categories if specified
    if category_filter:
        categories = {k: v for k, v in categories.items() if k in category_filter}
    
    if not categories:
        st.warning("No categories selected. Please select at least one category.")
        return [], []
    
    images_per_category = max(1, int(num_images / len(categories)))
    
    for category, directory in categories.items():
        if os.path.exists(directory):
            files = glob.glob(os.path.join(directory, "*.jpg"))
            if files:
                selected_files = random.sample(files, min(images_per_category, len(files)))
                for img_path in selected_files:
                    image_paths.append(img_path)
                    
                    # Binary label (0=Cover, 1=Stego)
                    binary_label = 0 if category == "Cover" else 1
                    
                    # Full classification info
                    ground_truth.append({
                        "path": img_path,
                        "category": category,
                        "binary_label": binary_label,
                        "multiclass_label": -1 if category == "Cover" else 
                                          0 if category == "JMiPOD" else
                                          1 if category == "JUNIWARD" else 2
                    })
    
    return image_paths, ground_truth


def process_image(image_path):
    """Load and preprocess an image"""
    try:
        image = Image.open(image_path).convert("RGB")
        processed_image = val_transform(image)
        return image, processed_image.unsqueeze(0)  # Add batch dimension
    except Exception as e:
        st.error(f"Error processing image {image_path}: {e}")
        return None, None


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


def display_confidence_gauge(probability, is_correct):
    """Display a colored confidence gauge for binary prediction"""
    fig, ax = plt.subplots(figsize=(6, 0.5))
    
    # Create a horizontal gradient colorbar
    gradient = np.linspace(0, 1, 100).reshape(1, -1)
    
    # Display the gradient
    ax.imshow(gradient, aspect='auto', cmap=custom_cmap)
    
    # Add a vertical line at the probability point
    ax.axvline(x=probability * 100, color='black', lw=3)
    
    # Add Cover and Stego labels
    ax.text(10, 0, "Cover", fontsize=10, va='center', ha='center', 
            color='white', fontweight='bold')
    ax.text(90, 0, "Stego", fontsize=10, va='center', ha='center', 
            color='white', fontweight='bold')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    return fig


def display_stego_type_chart(multiclass_probs):
    """Display a bar chart of steganography type probabilities"""
    fig, ax = plt.subplots(figsize=(6, 2.5))
    stego_types = ["JMiPOD", "JUNIWARD", "UERD"]
    
    # Create bar chart
    bars = ax.bar(stego_types, multiclass_probs, color=colors)
    
    # Add value labels on top of bars
    for bar, prob in zip(bars, multiclass_probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Customize appearance
    ax.set_ylim(0, 1.1)
    ax.set_title("Steganography Type Probabilities", fontsize=10)
    
    plt.tight_layout()
    return fig


def display_results_streamlit(image, results, ground_truth):
    """Display the results in a Streamlit-friendly way"""
    st.subheader("Prediction Results")
    
    col1, col2 = st.columns([2, 2])
    
    with col1:
        st.image(image, caption=f"Image: {os.path.basename(ground_truth['path'])}", use_container_width=True)
    
    with col2:
        st.write("##### Ground Truth:")
        st.info(f"**{ground_truth['category']}**")
        
        st.write("##### Prediction:")
        binary_result = "Stego" if results["binary_pred"] == 1 else "Cover"
        binary_correct = results["binary_pred"] == ground_truth["binary_label"]
        
        if binary_correct:
            st.success(f"**{binary_result}** ✓ CORRECT")
        else:
            st.error(f"**{binary_result}** ✗ INCORRECT")
        
        # Display confidence gauge
        st.write("##### Prediction Confidence:")
        confidence_fig = display_confidence_gauge(results["binary_prob"], binary_correct)
        st.pyplot(confidence_fig)
        st.caption(f"Confidence: {results['binary_prob']:.4f}")
        
        # Only display stego type information if predicted as stego
        if results["binary_pred"] == 1 and results["multiclass_probs"] is not None:
            st.write("##### Detected Steganography Type:")
            
            stego_types = ["JMiPOD", "JUNIWARD", "UERD"]
            detected_type = stego_types[results["multiclass_pred"]]
            
            if ground_truth["binary_label"] == 1:
                true_type = stego_types[ground_truth["multiclass_label"]]
                multiclass_correct = results["multiclass_pred"] == ground_truth["multiclass_label"]
                
                if multiclass_correct:
                    st.success(f"**{detected_type}** ✓ CORRECT")
                else:
                    st.error(f"**{detected_type}** ✗ INCORRECT (should be {true_type})")
            else:
                # If ground truth is cover but predicted as stego
                st.error(f"**{detected_type}** ✗ (Ground truth is Cover)")
            
            # Display steganography type probabilities chart
            st.pyplot(display_stego_type_chart(results["multiclass_probs"]))


def main():
    st.title("🔍 Steganalysis Interactive Demo")
    st.markdown("""
    This demo allows you to explore a steganalysis model that detects hidden messages in images.
    The model performs two tasks:
    1. **Binary Classification**: Determine if an image contains hidden data (Stego) or not (Cover)
    2. **Multi-class Classification**: Identify the steganography technique used (JMiPOD, JUNIWARD, or UERD)
    """)
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    # Get device
    device = DEVICE
    st.sidebar.info(f"Using device: {device}")
    
    # Get list of checkpoints
    available_checkpoints = get_available_checkpoints()
    if not available_checkpoints:
        st.error("No checkpoints found. Please train a model first.")
        st.stop()
    
    # Model selection
    checkpoint_options = [os.path.basename(cp) for cp in available_checkpoints]
    selected_checkpoint_name = st.sidebar.selectbox(
        "Select Model Checkpoint",
        checkpoint_options,
    )
    selected_checkpoint = available_checkpoints[checkpoint_options.index(selected_checkpoint_name)]
    
    # Load model
    with st.spinner("Loading model...", show_time=True):
        model = load_model(selected_checkpoint, device)
        
    if model is None:
        st.error("Failed to load model.")
        st.stop()
    else:
        st.sidebar.success("Model loaded successfully!")
    
    # Image selection options
    st.sidebar.subheader("Image Selection")
    
    # Category selection
    st.sidebar.write("Select image categories:")
    use_cover = st.sidebar.checkbox("Cover (no hidden data)", value=True)
    use_jmipod = st.sidebar.checkbox("JMiPOD", value=True)
    use_juniward = st.sidebar.checkbox("JUNIWARD", value=True) 
    use_uerd = st.sidebar.checkbox("UERD", value=True)
    
    categories = []
    if use_cover:
        categories.append("Cover")
    if use_jmipod:
        categories.append("JMiPOD")
    if use_juniward:
        categories.append("JUNIWARD")
    if use_uerd:
        categories.append("UERD")
    
    # Number of random images
    num_images = st.sidebar.slider("Number of images to analyze", 
                               min_value=1, max_value=12, value=4, step=1)
    
    # Run analysis button
    if st.sidebar.button("Run Analysis"):
        if not categories:
            st.warning("Please select at least one image category.")
            st.stop()
            
        with st.spinner("Getting random images..."):
            image_paths, ground_truths = get_random_images(
                BASE_DATA_DIR, 
                num_images=num_images,
                category_filter=categories
            )
            
        if not image_paths:
            st.error("No images found. Check that your data directories are correctly set.")
            st.stop()
        
        # Create a placeholder for results
        results_placeholder = st.empty()
        
        with results_placeholder.container():
            st.subheader("Analysis Results")
            
            # Create a layout for displaying multiple images
            cols = 2  # Number of columns
            rows = (len(image_paths) + cols - 1) // cols  # Calculate number of rows needed
            
            # Process each image in batches
            with st.spinner("Running inference..."):
                results_list = []
                
                for idx, (img_path, gt) in enumerate(zip(image_paths, ground_truths)):
                    # Load and process image
                    orig_image, processed_image = process_image(img_path)
                    if orig_image is None:
                        continue
                    
                    # Run inference
                    results = run_inference(model, processed_image, device)
                    results_list.append((orig_image, results, gt))
            
            # Display results in a grid layout
            for i in range(rows):
                row_cols = st.columns(cols)
                
                for j in range(cols):
                    idx = i * cols + j
                    if idx < len(results_list):
                        with row_cols[j]:
                            orig_image, results, gt = results_list[idx]
                            display_results_streamlit(orig_image, results, gt)
                            st.markdown("---")
    
    # Initial instructions
    else:
        st.info("👈 Select options in the sidebar and click 'Run Analysis' to start.")
        st.markdown("""
        ### How It Works
        
        1. Select a model checkpoint from the sidebar
        2. Choose which image categories to analyze
        3. Set the number of random images to process
        4. Click "Run Analysis"
        
        The app will then show:
        - The original images
        - Ground truth labels
        - Model predictions
        - Confidence scores
        - Steganography technique detection (if applicable)
        """)


if __name__ == "__main__":
    main()