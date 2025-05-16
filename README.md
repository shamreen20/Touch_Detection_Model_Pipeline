# Touch Detection Model Pipeline
This repository contains a PyTorch-based pipeline for training a touch detection model using tactile images from a robotic grasping dataset. The pipeline processes tactile images (`touch_middle_3.png` and `touch_thumb_3.png`) and grasp success labels from the dataset described in Learning Gentle Grasping Using Vision, Sound, and Touch. It includes a custom CNN, an MLP, and an optional PyTouch baseline.

## Dataset
The dataset consists of 1,500 grasping trials, stored in the collected_data folder. Each trial is in a subdirectory <b>(e.g., data_2024_07_13_114803)</b> containing:

- **Tactile Images:** touch_middle_3.png and touch_thumb_3.png (RGB images from DIGIT sensors, lifting phase).

- **Labels:** labels_supervised.npy (grasp success label, binary: 1/0, in a 1D or 2D array).

- **Total:** ~3,000 input-label pairs (1,500 trials × 2 images per trial).

Download the dataset from OPARA and extract it to `./dataset_gentle_grasping/collected_data.`

## Requirements

- Python 3.8+
- Conda (recommended for virtual environment)
- PyTorch, torchvision, numpy, pillow
- (Optional) PyTouch for baseline

## Installation

1. **Create a virtual environment:**
```bash
 conda create -n touch_detection python=3.8
 conda activate touch_detection
 ```


2. **Install dependencies:**
```bash
pip install torch torchvision numpy pillow
```

For the optional PyTouch baseline:
```bash
pip install git+https://github.com/facebookresearch/PyTouch.git
```


3. **Set up the dataset:**

- Download and extract the dataset to `./dataset_gentle_grasping/collected_data.`
- Ensure each trial subdirectory contains touch_middle_3.png, touch_thumb_3.png, and labels_supervised.npy.



## Usage
The pipeline is implemented in `touch_detection_pipeline.py`, which:

- Loads tactile images and grasp success labels.
- Trains a custom CNN and MLP for touch detection.
- Attempts a PyTouch baseline (if installed).
- Splits data into 80/20 train/validation sets.
- Trains models for 10 epochs and reports accuracy.

### Running the Pipeline

1. Update the `data_dir` variable in `touch_detection_pipeline.py` to your `collected_data` path (e.g., `./dataset_gentle_grasping/collected_data)`.

2. Run the script:
```
python touch_detection_pipeline.py
```

3. Output:
- Number of loaded image-label pairs (~3,000).
- Training loss and accuracy per epoch.
- Validation accuracy per epoch.
- Warnings for missing files.
- PyTouch baseline status (if applicable).



## Pipeline Details

**Data Loading:**
- Images are resized to 128x128 and normalized.
- Labels are extracted from labels_supervised.npy (handles 1D or 2D arrays).


**Models:**
- **CNN:** 3 convolutional layers (32→64→128 channels, 3x3 kernels), max-pooling, and fully connected layers (512→1, sigmoid).
- **MLP:** Flattens images, processes through dense layers (1281283→1024→512→1, sigmoid).
- **PyTouch:** Baseline using TouchDetect task (may require custom integration; skips if unavailable).


**Training:**
- Batch size: 32
- Optimizer: Adam (lr=0.001)
- Loss: Binary Cross-Entropy
- Epochs: 10
- Device: CUDA (if available) or CPU



## Troubleshooting

**Label Shape Error:** If `labels_supervised.npy` has an unexpected shape, the script prints warnings. Verify shapes with:
```
import numpy as np
import os
for subdir in os.listdir('./dataset_gentle_grasping/collected_data'):
    label_file = os.path.join('./dataset_gentle_grasping/collected_data', subdir, 'labels_supervised.npy')
    if os.path.exists(label_file):
        print(f"{label_file}: {np.load(label_file).shape}")
```

- **Missing Files:** Check console warnings for missing images or labels.
- **PyTouch Issues:** If PyTouch fails, the script skips it. Ensure proper installation or contact PyTouch maintainers.
- **Dataset Path:** Update data_dir if the dataset is not in ./dataset_gentle_grasping/collected_data.

## Customization

- **Model Variants:** To use DenseNet or ResNet, modify `TouchCNN` (e.g., `torchvision.models.resnet18(pretrained=True)`).
- **Hyperparameters:** Adjust `num_epochs`, `batch_size`, or learning rate in `train_model`.
- **Image Size:** Change `transforms.Resize` if DIGIT images require a different resolution.

## Contact
For dataset-related questions, contact `shamreen.tabassum@mailbox.tu-dresden.de` For pipeline issues, refer to the dataset's README or seek further assistance.

