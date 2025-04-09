# ALASKA2 Multi-Task Steganalysis

This repository contains a multi-task deep learning approach for the ALASKA2 Image Steganalysis competition. The model is designed to:

1. Detect if an image has hidden data (cover vs stego)
2. Identify which steganography algorithm was used (JMiPOD, JUNIWARD, or UERD)

## Project Structure

The project is organized into modular components:

- `config.py`: Configuration parameters, constants, shared imports
- `model.py`: Multi-task ResNet18 model definition with checkpoint functionality
- `dataset.py`: Data loading and preparation utilities
- `train_utils.py`: Training, evaluation, and metrics calculation functions
- `train.py`: Main training script with checkpointing and resumption functionality
- `predict.py`: Script for generating predictions on test data

## Installation

### Requirements

- Python 3.6+
- PyTorch 1.7+
- torchvision
- scikit-learn
- pandas
- numpy
- tqdm

### Setup

```bash
# Clone the repository
git clone [repository_url]
cd alaska2-steganalysis

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

The code expects the ALASKA2 dataset to be organized in the following structure:

```
input/alaska2-image-steganalysis/
├── Cover/
│   ├── 00000.jpg
│   ├── 00001.jpg
│   └── ...
├── JMiPOD/
│   ├── 00000.jpg
│   └── ...
├── JUNIWARD/
│   ├── 00000.jpg
│   └── ...
├── UERD/
│   ├── 00000.jpg
│   └── ...
└── Test/
    ├── 00000.jpg
    └── ...
```

You can modify the data paths in `config.py` if your data is stored differently.

## Training

### Basic Training

To start training from scratch:

```bash
python train.py --epochs 10 --batch_size 64 --lr 1e-4
```

### Training Options

```
usage: train.py [-h] [--batch_size BATCH_SIZE] [--lr LR] [--epochs EPOCHS]
                [--mc_weight MC_WEIGHT] [--checkpoint CHECKPOINT]
                [--checkpoint_freq CHECKPOINT_FREQ] [--data_dir DATA_DIR]
                [--val_split VAL_SPLIT] [--seed SEED]

Train Multi-Task Steganalysis Model

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch size for training
  --lr LR               Learning rate
  --epochs EPOCHS       Number of epochs to train
  --mc_weight MC_WEIGHT
                        Weight for multi-class loss
  --checkpoint CHECKPOINT
                        Path to checkpoint to resume training from
  --checkpoint_freq CHECKPOINT_FREQ
                        Save checkpoint every N epochs
  --data_dir DATA_DIR   Base directory for ALASKA2 dataset
  --val_split VAL_SPLIT
                        Validation split ratio (0-1)
  --seed SEED           Random seed for reproducibility
```

### Checkpoint System

The training script automatically saves:
- Regular checkpoints at intervals specified by `--checkpoint_freq`
- The best model based on validation AUC
- A final model after training completes

All checkpoints are saved to the `./checkpoints/` directory by default.

### Resuming Training

To resume training from a checkpoint:

```bash
python train.py --checkpoint ./checkpoints/checkpoint_epoch_5.pth --epochs 5
```

This will load the model, optimizer state, and training history from the checkpoint and continue training for 5 more epochs.

## Making Predictions

To generate predictions on the test set:

```bash
python predict.py --checkpoint ./checkpoints/best_model_auc_0.9123.pth --output submission.csv
```

### Prediction Options

```
usage: predict.py [-h] --checkpoint CHECKPOINT [--test_dir TEST_DIR]
                  [--output OUTPUT] [--batch_size BATCH_SIZE]

Generate Predictions with Trained Model

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT
                        Path to trained model checkpoint
  --test_dir TEST_DIR   Directory containing test images
  --output OUTPUT       Output file path for predictions
  --batch_size BATCH_SIZE
                        Batch size for prediction
```

## Model Architecture

The model uses a pre-trained ResNet18 backbone with two task-specific heads:
1. Binary classification head for stego detection (cover vs. stego)
2. Multi-class classification head for stego algorithm identification (JMiPOD, JUNIWARD, UERD)

The multi-task approach allows the model to leverage information from both tasks to improve overall performance.

## Acknowledgments

This project is built for the [ALASKA2 Image Steganalysis](https://www.kaggle.com/c/alaska2-image-steganalysis/) competition on Kaggle.

## License

[MIT License](LICENSE)