# Crack Detection using CNN (Grayscale Image Processing)

This repository contains a crack / no-crack image classification pipeline using a Convolutional Neural Network (CNN).
The workflow standardizes input images using grayscale preprocessing and trains a regularized CNN model to improve
generalization on unseen structural surface images (e.g., bridges, walls, high-rise buildings).

## Project Overview

- Task: Binary image classification (Crack vs No_Crack)
- Input: 224 x 224 grayscale images (single channel)
- Output: Probability score and class label (Crack / No_Crack)
- Framework: TensorFlow / Keras

## Key Features

- Grayscale preprocessing to emphasize texture/edges and reduce input complexity
- Image resizing to 224 x 224
- Pixel normalization (0–1 scaling)
- Automatic 80/20 training-validation split using Keras ImageDataGenerator
- Deep CNN feature extractor using stacked Conv2D + MaxPooling
- Regularization to reduce overfitting:
  - Dropout layers
  - L1/L2 regularization on dense layers
- Training stability and model selection:
  - ReduceLROnPlateau to reduce learning rate when validation loss plateaus
  - ModelCheckpoint to save the best model based on validation loss
- Training curves for accuracy and loss visualization

## Dataset Structure

Keras expects the dataset to be organized into subfolders (one per class). Example:

Dataset/
  Crack/
    img1.jpg
    img2.jpeg
    ...
  No_Crack/
    img3.jpg
    img4.jpeg
    ...

Supported formats include .jpg, .jpeg, and .png.

## Preprocessing Pipeline

1. Load images from directory
2. Convert to grayscale (single channel)
3. Resize to 224 x 224
4. Normalize pixels to [0, 1]
5. Split into training (80%) and validation (20%)

## Model Summary

- Convolution blocks for hierarchical feature extraction
- Dense layers with regularization for classification
- Sigmoid output neuron for binary prediction

Compiled with:
- Optimizer: Adam
- Loss: Binary Crossentropy
- Metric: Accuracy

## Training Strategy

- Batch size: 32
- Epochs: 50
- Callbacks:
  - ReduceLROnPlateau (monitor: val_loss)
  - ModelCheckpoint (save_best_only: True, monitor: val_loss)

The best model is saved as:
- best_model.h5

## How to Run

1. Install dependencies:
   pip install tensorflow matplotlib pillow

2. Update the dataset path in the script:
   Example (Windows):
   r"C:\Users\gkash\Downloads\Pera\FYP\Test IP"

3. Run training:
   Execute the Python script or notebook containing the training code.

## Outputs

- best_model.h5 (best performing model on validation loss)
- Training plots:
  - Accuracy vs Epochs
  - Loss vs Epochs

## Applications

- Structural health monitoring
- Automated inspection workflows
- Remote crack assessment in hard-to-reach infrastructure
- Preventive maintenance support systems

## Author

Ashwin
Image Processing and CNN Model Development
