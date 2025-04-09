#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model definition for Multi-Task Learning model for ALASKA2 Image Steganalysis.
"""

import torch
import torch.nn as nn
from torchvision import models


class MultiTaskResNet(nn.Module):
    """
    Multi-Task ResNet model for ALASKA2 Steganalysis.
    Uses a pre-trained ResNet backbone and two task-specific heads.
    - Binary Head: Predicts Cover (0) vs. Stego (1).
    - Multi-class Head: Predicts Stego Algorithm (JMiPOD/JUNIWARD/UERD).
    
    Supports different ResNet backbones: 18, 34, 50, 101, 152
    """

    def __init__(self, num_classes_multiclass=3, backbone_type=18, pretrained=True):
        super(MultiTaskResNet, self).__init__()
        
        # Convert string backbone type to int if needed (for backward compatibility)
        if isinstance(backbone_type, str):
            if backbone_type.startswith('resnet'):
                try:
                    backbone_type = int(backbone_type[6:])
                except ValueError:
                    pass
        
        # Select the appropriate ResNet backbone
        if backbone_type == 18 or backbone_type == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        elif backbone_type == 34 or backbone_type == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
        elif backbone_type == 50 or backbone_type == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        elif backbone_type == 101 or backbone_type == "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet101(weights=weights)
        elif backbone_type == 152 or backbone_type == "resnet152":
            weights = models.ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet152(weights=weights)
        else:
            raise ValueError(f"Unsupported ResNet backbone type: {backbone_type}")
            
        # Store the backbone type in a normalized format (integer)
        if isinstance(backbone_type, str) and backbone_type.startswith('resnet'):
            self.backbone_type = int(backbone_type[6:])
        else:
            self.backbone_type = int(backbone_type)
            
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove original classifier

        # Binary Classification Head (Cover=0, Stego=1)
        self.binary_head = nn.Linear(num_ftrs, 1)

        # Multi-class Classification Head (JMiPOD=0, JUNIWARD=1, UERD=2)
        self.multiclass_head = nn.Linear(num_ftrs, num_classes_multiclass)

    def forward(self, x):
        features = self.backbone(x)
        binary_output_logits = self.binary_head(features)
        multiclass_output_logits = self.multiclass_head(features)
        return binary_output_logits, multiclass_output_logits

    def save_checkpoint(
        self,
        path,
        optimizer=None,
        scheduler=None,
        epoch=0,
        best_val_auc=0.0,
        train_history=None,
        val_history=None,
    ):
        """
        Save model checkpoint with additional training information.

        Args:
            path (str): Path where to save the checkpoint
            optimizer (torch.optim): Optimizer state
            scheduler: Learning rate scheduler state
            epoch (int): Current epoch number
            best_val_auc (float): Best validation AUC score
            train_history (dict): Training metrics history
            val_history (dict): Validation metrics history
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
            "best_val_auc": best_val_auc,
            "backbone_type": self.backbone_type,  # Store backbone type
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if train_history is not None:
            checkpoint["train_history"] = train_history

        if val_history is not None:
            checkpoint["val_history"] = val_history

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path, device, optimizer=None, scheduler=None):
        """
        Load model checkpoint and return model and training information.

        Args:
            path (str): Path to the saved checkpoint
            device (torch.device): Device to load the model on
            optimizer (torch.optim, optional): Optimizer to load state into
            scheduler (optional): Learning rate scheduler to load state into

        Returns:
            tuple: (model, optimizer, scheduler, epoch, best_val_auc, train_history, val_history)
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Extract model architecture parameters from the state dict
        num_multiclass = None
        for key in checkpoint["model_state_dict"].keys():
            if "multiclass_head" in key and "weight" in key:
                # Get output size of multiclass head
                num_multiclass = checkpoint["model_state_dict"][key].size(0)
                break

        # Get backbone type from checkpoint, default to 18 for backward compatibility
        backbone_type = checkpoint.get("backbone_type", 18)

        # Create model with the same architecture
        model = cls(num_classes_multiclass=num_multiclass, backbone_type=backbone_type)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        epoch = checkpoint.get("epoch", 0)
        best_val_auc = checkpoint.get("best_val_auc", 0.0)
        train_history = checkpoint.get("train_history", {})
        val_history = checkpoint.get("val_history", {})

        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if provided
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return (
            model,
            optimizer,
            scheduler,
            epoch,
            best_val_auc,
            train_history,
            val_history,
        )

# For backward compatibility
MultiTaskResNet18 = MultiTaskResNet
