# Single Neuron From Scratch

A simple implementation of a single artificial neuron using Python and NumPy.

This project was created to understand the fundamental concepts behind neural networks without using deep learning frameworks such as TensorFlow or PyTorch.

## What I Learned

Through this project, I learned about:

* Inputs and features
* Weights
* Bias
* Weighted sum
* Sigmoid activation function
* Forward propagation
* Loss calculation
* Gradient descent
* Weight and bias updates
* Epochs and learning rate
* Training a neuron
* Making predictions on new data

## Project Idea

The neuron predicts whether a student will **Pass or Fail** based on:

* Study hours
* Attendance percentage

The model uses two input features:

```text
Study Hours -------- Weight 1 ----\
                                   > Neuron → Sigmoid → Pass / Fail
Attendance --------- Weight 2 ----/
                         +
                        Bias
```

## Neural Network Calculation

The neuron first calculates the weighted sum:

```text
z = x1*w1 + x2*w2 + bias
```

The result is passed through the sigmoid activation function:

```text
sigmoid(z) = 1 / (1 + e^(-z))
```

The sigmoid output is between `0` and `1`.

* Output >= 0.5 → Pass
* Output < 0.5 → Fail

## Training

The neuron starts with random weights.

During training, it repeatedly performs:

```text
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Calculate Error/Loss
  ↓
Calculate Gradients
  ↓
Update Weights and Bias
  ↓
Repeat
```

Gradient descent is used to update the weights and bias so that the prediction error decreases.

## Technologies Used

* Python
* NumPy

## Run the Project

Install the required dependency:

```bash
pip install -r requirements.txt
```

Run the program:

```bash
python main.py
```

Enter the study hours and attendance percentage when prompted.

Example:

```text
Enter study hours: 6
Enter attendance percentage: 75

Pass probability: 0.86
Prediction: PASS
```

## Purpose

The main purpose of this project is not to build a highly accurate student prediction system. It is a learning project designed to understand how a basic artificial neuron works internally before moving to multi-layer neural networks and deep learning frameworks.
