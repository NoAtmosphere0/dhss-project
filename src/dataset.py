#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset and data loading utilities for ALASKA2 Image Steganalysis.
"""

from config import *


class Alaska2Dataset(Dataset):
    """
    Dataset class for ALASKA2 Image Steganalysis competition.
    Reads images from specified directories and assigns labels accordingly.
    Can be initialized directly with file paths and labels for train/val splits.
    """

    def __init__(self, image_paths, labels=None, transform=None, is_test_set=False):
        """
        Args:
            image_paths (list): List of full paths to image files.
            labels (list, optional): List of dictionaries [{'binary': b, 'multiclass': mc}, ...].
                                     Required if is_test_set=False.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test_set (bool): If True, __getitem__ returns image and image_id.
                                If False, returns image, binary_label, multiclass_label.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_test_set = is_test_set

        if not is_test_set and labels is None:
            raise ValueError(
                "Labels must be provided for training/validation datasets."
            )
        if not is_test_set and len(image_paths) != len(labels):
            raise ValueError(
                "Number of image paths and labels must match for train/val."
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            # Ensure image is loaded in RGB format
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a placeholder if image loading fails
            image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color="red")
            if self.is_test_set:
                # Need to return something matching the expected test output format
                image_id = os.path.basename(img_path)
                if self.transform:
                    image = self.transform(image)
                return image, image_id
            else:
                # Return dummy labels matching expected train/val output format
                binary_label = torch.tensor(0, dtype=torch.float32)
                multiclass_label = torch.tensor(-1, dtype=torch.long)
                if self.transform:
                    image = self.transform(image)
                return image, binary_label, multiclass_label

        if self.transform:
            image = self.transform(image)

        if self.is_test_set:
            image_id = os.path.basename(img_path)  # Extract filename as ID
            return image, image_id
        else:
            label_dict = self.labels[idx]
            binary_label = torch.tensor(label_dict["binary"], dtype=torch.float32)
            # Use long for CrossEntropyLoss index, ensure multiclass label is valid index or ignored
            multiclass_label = torch.tensor(label_dict["multiclass"], dtype=torch.long)
            return image, binary_label, multiclass_label


def prepare_data_splits(base_dir, val_split=0.1, random_seed=42):
    """
    Scans ALASKA2 directories, creates stratified train/validation splits.
    Returns:
        tuple: (train_paths, train_labels, val_paths, val_labels)
    """
    all_paths = []
    all_labels = []  # List of dictionaries

    folders = {
        "Cover": COVER_DIR,
        "JMiPOD": JMIPOD_DIR,
        "JUNIWARD": JUNIWARD_DIR,
        "UERD": UERD_DIR,
    }
    multiclass_map = {"JMiPOD": 0, "JUNIWARD": 1, "UERD": 2}  # Stego algorithm index

    print("Scanning image folders...")
    for label_name, folder_path in folders.items():
        print(f"  Scanning {label_name} folder: {folder_path}")
        # Use glob to find all .jpg files (ALASKA2 uses JPG)
        image_files = glob.glob(os.path.join(folder_path, "*.jpg"))
        print(f"    Found {len(image_files)} images.")

        for img_path in image_files:
            all_paths.append(img_path)
            if label_name == "Cover":
                binary_label = 0
                multiclass_label = -1  # Not applicable for Cover
            else:
                binary_label = 1  # It's a stego image
                multiclass_label = multiclass_map[label_name]

            all_labels.append({"binary": binary_label, "multiclass": multiclass_label})

    if not all_paths:
        raise FileNotFoundError(
            f"No images found in the specified directories under {base_dir}. Check paths."
        )

    print(f"Total training/validation images found: {len(all_paths)}")

    # Create a combined label for stratification (0=Cover, 1=JMiPOD, 2=JUNIWARD, 3=UERD)
    stratify_labels = [
        l["binary"] * (l["multiclass"] + 1) if l["binary"] == 1 else 0
        for l in all_labels
    ]

    print("Splitting data into training and validation sets...")
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths,
        all_labels,
        test_size=val_split,
        random_state=random_seed,
        stratify=stratify_labels,  # Ensure proportion of Cover/JMiPOD/JUNIWARD/UERD is similar in both sets
    )
    print(f"  Training set size: {len(train_paths)}")
    print(f"  Validation set size: {len(val_paths)}")

    return train_paths, train_labels, val_paths, val_labels


def create_data_loaders(
    train_paths, train_labels, val_paths, val_labels, batch_size=64
):
    """
    Create PyTorch DataLoaders for training and validation.

    Args:
        train_paths (list): List of training image paths
        train_labels (list): List of training labels
        val_paths (list): List of validation image paths
        val_labels (list): List of validation labels
        batch_size (int): Batch size

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_dataset = Alaska2Dataset(train_paths, train_labels, transform=train_transform)
    val_dataset = Alaska2Dataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
    )

    return train_loader, val_loader


def create_test_loader(test_dir, batch_size=64):
    """
    Create PyTorch DataLoader for test data.

    Args:
        test_dir (str): Directory containing test images
        batch_size (int): Batch size

    Returns:
        DataLoader: Test data loader
    """
    test_image_paths = glob.glob(os.path.join(test_dir, "*.jpg"))
    if not test_image_paths:
        raise FileNotFoundError(f"No test images found in {test_dir}")

    print(f"Found {len(test_image_paths)} test images.")
    test_dataset = Alaska2Dataset(
        image_paths=test_image_paths,
        labels=None,
        transform=test_transform,
        is_test_set=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
    )

    return test_loader
