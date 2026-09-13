# Backpropagation Neural Network from Scratch

This project implements a simple **feedforward neural network with backpropagation from scratch using NumPy**.

The neural network is trained to solve the **XOR problem**. The main goal of this project is to understand how forward propagation, loss calculation, backpropagation, gradient calculation, and weight updates work internally without using deep learning frameworks such as TensorFlow or PyTorch.

## Project Structure

```text
05-backpropagation/
│
├── main.py
├── neural_network.py
├── requirements.txt
├── README.md
│
└── results/
    └── loss_curve.png
```

## XOR Dataset

The XOR problem is used as the training dataset.

| X1 | X2 | Output |
| -- | -- | ------ |
| 0  | 0  | 0      |
| 0  | 1  | 1      |
| 1  | 0  | 1      |
| 1  | 1  | 0      |

XOR is not linearly separable, so a simple single-layer perceptron cannot solve it. A neural network with a hidden layer can learn the XOR relationship.

## Neural Network Architecture

The network contains:

```text
Input Layer
    2 neurons
       ↓
Hidden Layer
    2 neurons
       ↓
Output Layer
    1 neuron
```

Architecture:

```text
x1 ─────┐
        ├──> Hidden Neurons ───> Output
x2 ─────┘
```

The **Sigmoid activation function** is used in the hidden and output layers.

## How Backpropagation Works

Training follows these main steps:

```text
Input Data
    ↓
Forward Propagation
    ↓
Prediction
    ↓
Loss Calculation
    ↓
Backpropagation
    ↓
Gradient Calculation
    ↓
Update Weights and Biases
    ↓
Repeat for Multiple Epochs
```

### 1. Forward Propagation

The input is passed through the network.

For the hidden layer:

```text
Z1 = XW1 + b1
A1 = sigmoid(Z1)
```

For the output layer:

```text
Z2 = A1W2 + b2
Output = sigmoid(Z2)
```

### 2. Loss Calculation

Mean Squared Error is used to measure the difference between the expected output and predicted output.

```text
Loss = mean((y - prediction)²)
```

A smaller loss means the predictions are getting closer to the expected outputs.

### 3. Backpropagation

Backpropagation calculates how much each weight contributed to the prediction error.

The error is propagated backward:

```text
Output Error
     ↓
Output Layer Gradients
     ↓
Hidden Layer Error
     ↓
Hidden Layer Gradients
```

### 4. Weight Update

Gradient descent updates the weights.

```text
new_weight = old_weight - learning_rate × gradient
```

The same process is used to update the biases.

Repeating this process allows the neural network to gradually reduce its loss.

## Technologies Used

* Python
* NumPy
* Matplotlib

No TensorFlow, PyTorch, or Scikit-learn is used for implementing the neural network.

## Installation

Clone the repository and move into the project directory.

Install the required packages:

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install numpy matplotlib
```

## Running the Project

Run:

```bash
python main.py
```

The program will:

1. Create the XOR dataset.
2. Initialize the neural network.
3. Perform forward propagation.
4. Calculate the loss.
5. Perform backpropagation.
6. Update weights and biases.
7. Train for multiple epochs.
8. Display the final predictions.
9. Save the training loss graph.

## Expected Output

After training, the network should learn the XOR relationship.

```text
Input: [0 0] -> Prediction: 0
Input: [0 1] -> Prediction: 1
Input: [1 0] -> Prediction: 1
Input: [1 1] -> Prediction: 0
```

The exact loss and learned weight values may vary depending on weight initialization.

## Training Loss

During training, the loss should gradually decrease.

The generated loss curve is saved at:

```text
results/loss_curve.png
```

![Training Loss Curve](results/loss_curve.png)

The decreasing loss demonstrates that gradient descent and backpropagation are successfully adjusting the network parameters.

## Concepts Learned

Through this project, I implemented and understood:

* Feedforward Neural Networks
* Forward Propagation
* Hidden Layers
* Sigmoid Activation Function
* Mean Squared Error
* Backpropagation
* Chain Rule
* Gradient Calculation
* Gradient Descent
* Weight Updates
* Bias Updates
* Learning Rate
* Epochs
* XOR Classification
* Training Loss Visualization

## Key Learning

The main idea behind backpropagation is:

> Calculate how much each parameter contributed to the final error, propagate that error backward through the network, and use the calculated gradients to update the parameters.

In simple terms:

```text
Predict → Calculate Error → Go Backward → Update Weights → Repeat
```

This project provides a basic understanding of the learning mechanism used by neural networks before moving to larger deep learning architectures.
