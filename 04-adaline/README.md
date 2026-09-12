# ADALINE (Adaptive Linear Neuron)

This project implements an **ADALINE (Adaptive Linear Neuron)** from scratch using Python and NumPy.

The main goal of this project is to understand how ADALINE performs binary classification using a linear neuron, Mean Squared Error, and Gradient Descent.

## Concepts Covered

* Adaptive Linear Neuron (ADALINE)
* Weighted sum and bias
* Linear output
* Mean Squared Error (MSE)
* Gradient Descent
* Weight and bias updates
* Feature standardization
* Binary classification
* Loss visualization
* Decision boundary

## ADALINE Model

For input features \(X\), weights \(w\), and bias \(b\), the linear output is:

$$
z = Xw + b
$$

The error is calculated as:

$$
e = y-z
$$

The loss function used during training is:

$$
J = \frac{1}{2n}\sum_{i=1}^{n}(y_i-z_i)^2
$$

Gradient descent is then used to update the weights and bias to reduce this loss.

## Dataset

A small manually created dataset is used to understand the working of ADALINE clearly.

```python
X = np.array([
    [1, 2],
    [2, 1],
    [2, 3],
    [3, 2],
    [-1, -2],
    [-2, -1],
    [-2, -3],
    [-3, -2]
])

y = np.array([
    1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    -1
])
```

The problem contains two classes:

* `+1`
* `-1`

This makes it a binary classification problem.

## Project Structure

```text
04-adaline/
│
├── adaline.py
├── main.py
├── requirements.txt
└── README.md
```

## Implementation

### `adaline.py`

Contains the ADALINE implementation from scratch, including:

* Weight initialization
* Linear output calculation
* Error calculation
* MSE loss calculation
* Gradient calculation
* Weight updates
* Bias updates
* Binary prediction

### `main.py`

Handles:

* Dataset creation
* Feature standardization
* Model initialization
* Model training
* Prediction
* Accuracy calculation
* Loss visualization
* Decision boundary visualization

## Feature Standardization

Before training, the input features are standardized using:

$$
x'=\frac{x-\mu}{\sigma}
$$

where:

* \(\mu\) = mean of the feature
* \(\sigma\) = standard deviation of the feature

Standardization helps Gradient Descent learn more effectively when features have different numerical scales.

## Prediction

After training, the linear output is converted into a class using a threshold:

$$
\hat{y}=
\begin{cases}
+1, & z\geq0\\
-1, & z<0
\end{cases}
$$

## Visualizations

The project generates two visualizations.

### Loss Curve

Shows how the Mean Squared Error changes as the number of training epochs increases.

A decreasing loss indicates that the ADALINE model is learning.

### Decision Boundary

Shows the linear boundary learned by ADALINE to separate the `+1` and `-1` classes.

The decision boundary is defined by:

$$
w_1x_1+w_2x_2+b=0
$$

## Requirements

* Python
* NumPy
* Matplotlib

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

## Technologies Used

* **Python** — programming language
* **NumPy** — numerical operations and ADALINE implementation
* **Matplotlib** — loss and decision-boundary visualization

## Learning Outcomes

Through this project, I learned:

* How ADALINE works mathematically
* How ADALINE differs from a Perceptron
* How Mean Squared Error measures model error
* How Gradient Descent updates weights and bias
* How feature standardization affects training
* How a linear decision boundary separates two classes
* How to implement a basic neural network model from scratch using NumPy
