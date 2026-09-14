# CNN From Scratch – Vertical vs Horizontal Line Classification

A small Convolutional Neural Network project built from scratch using Python and NumPy.

The purpose of this project is to understand the internal working process of a CNN without depending on high-level deep-learning frameworks such as TensorFlow or PyTorch.

## Project Overview

The model classifies simple 5×5 binary images into two classes:

* Vertical Line
* Horizontal Line

The project demonstrates the major stages of a Convolutional Neural Network:

```text
Input Image
    ↓
Convolution
    ↓
Feature Map
    ↓
ReLU
    ↓
Max Pooling
    ↓
Flatten
    ↓
Dense Layer
    ↓
Softmax
    ↓
Cross-Entropy Loss
    ↓
Prediction
```

## Concepts Implemented

### 1. Convolution

A 3×3 kernel moves across the input image and performs element-wise multiplication followed by summation.

This produces a feature map.

### 2. Multiple Kernels

Two manually defined kernels are used:

* Vertical edge kernel
* Horizontal edge kernel

Each kernel extracts different information from the input image.

### 3. ReLU Activation

ReLU is applied using:

```text
ReLU(x) = max(0, x)
```

Negative values are converted to zero while positive values are retained.

### 4. Max Pooling

A 2×2 max-pooling operation reduces the size of the feature maps while retaining strong activations.

### 5. Flattening

The pooled feature maps are converted from matrices into one-dimensional vectors.

The features produced by multiple kernels are then combined.

### 6. Dense Layer

The extracted CNN features are passed into a fully connected layer containing two outputs:

```text
0 → Vertical Line
1 → Horizontal Line
```

### 7. Softmax

Softmax converts the output scores of the dense layer into class probabilities.

### 8. Cross-Entropy Loss

Cross-entropy measures how far the predicted probabilities are from the correct class.

A smaller loss indicates a better prediction.

### 9. Backpropagation and Gradient Descent

The gradients of the dense-layer weights and biases are calculated and the parameters are updated using gradient descent.

In this version, the convolution kernels are manually defined and remain fixed during training.

Only the dense-layer weights and biases are learned.

## Project Structure

```text
07-cnn-from-scratch/
│
├── main.py
├── convolution.py
├── pooling.py
├── functions.py
└── README.md
```

### `main.py`

Contains:

* Dataset
* Kernels
* Feature extraction
* Dense layer
* Training loop
* Backpropagation
* Testing and prediction

### `convolution.py`

Contains the manual implementation of the convolution operation.

### `pooling.py`

Contains the manual implementation of max pooling.

### `functions.py`

Contains:

* Softmax
* Cross-entropy loss

## Technologies Used

* Python
* NumPy

No TensorFlow or PyTorch is used in this project.

## Learning Outcome

Through this project, I learned how data flows through a basic CNN:

```text
Image
→ Convolution
→ ReLU
→ Pooling
→ Flatten
→ Dense Layer
→ Softmax
→ Loss
→ Backpropagation
```

I also learned the difference between feature extraction and classification.

The convolution kernels perform feature extraction, while the dense layer uses those extracted features to perform classification.

## Current Limitation

The convolution kernels are manually defined and are not updated during training.

Therefore, this project demonstrates the fundamental CNN pipeline and dense-layer learning rather than a fully trainable CNN.

A future extension of this project would implement backpropagation through the convolution layer so that the network learns its own kernels automatically.
