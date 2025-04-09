#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration file for Multi-Task Learning model for ALASKA2 Image Steganalysis.
Contains shared imports and configuration variables.
"""

import argparse
import glob
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# --- Configuration ---
# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Paths
BASE_DATA_DIR = "../input/alaska2-image-steganalysis"
COVER_DIR = os.path.join(BASE_DATA_DIR, "Cover")
JMIPOD_DIR = os.path.join(BASE_DATA_DIR, "JMiPOD")
JUNIWARD_DIR = os.path.join(BASE_DATA_DIR, "JUNIWARD")
UERD_DIR = os.path.join(BASE_DATA_DIR, "UERD")
TEST_DIR = os.path.join(BASE_DATA_DIR, "Test")

# Checkpoints directory
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 5
NUM_CLASSES_MULTICLASS = 3  # JMiPOD, JUNIWARD, UERD
MULTICLASS_LOSS_WEIGHT = 1.0
IMG_SIZE = 256
VAL_SPLIT = 0.1
RANDOM_SEED = 42

# Label mappings
LABEL_MAP = {"Cover": 0, "JMiPOD": 1, "JUNIWARD": 2, "UERD": 3}
CLASS_NAMES_BINARY = {0: "Cover", 1: "Stego"}
CLASS_NAMES_MULTICLASS = {0: "JMiPOD", 1: "JUNIWARD", 2: "UERD"}


# Set random seeds for reproducibility
def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


# Transformations
train_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = val_transform
