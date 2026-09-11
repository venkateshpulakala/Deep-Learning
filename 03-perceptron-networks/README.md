# Perceptron Loan Approval Classifier

A simple implementation of a **Perceptron from scratch using Python and NumPy**.

This project demonstrates how a perceptron learns to perform binary classification. A small loan approval dataset is used where the model predicts whether a loan should be **Approved** or **Rejected** based on the applicant's income and credit score.

## Project Objective

The main objective of this project is to understand the fundamental working of a perceptron, including:

* Inputs, weights, and bias
* Weighted sum
* Step activation function
* Perceptron learning rule
* Weight and bias updates
* Feature scaling
* Binary classification
* Model accuracy
* Decision boundary visualization

## Dataset

A small manually created dataset is currently used for learning purposes.

The model uses two input features:

| Feature      | Description              |
| ------------ | ------------------------ |
| Income       | Applicant's income       |
| Credit Score | Applicant's credit score |

The target variable represents the loan decision:

```text
0 = Loan Rejected
1 = Loan Approved
```

Example:

```text
Income    Credit Score    Loan Status
20        400             0
25        450             0
30        500             0
35        520             0
40        580             1
50        650             1
60        700             1
70        750             1
```

## How the Perceptron Works

The perceptron first calculates the weighted sum of the input features:

```text
z = w1*x1 + w2*x2 + b
```

where:

* `x1`, `x2` are input features
* `w1`, `w2` are weights
* `b` is the bias
* `z` is the weighted sum

### Step Activation Function

The weighted sum is passed through a step activation function:

```text
if z >= 0:
    prediction = 1
else:
    prediction = 0
```

Therefore:

```text
1 → Loan Approved
0 → Loan Rejected
```

## Perceptron Learning Rule

The prediction is compared with the actual value.

```text
error = actual - predicted
```

The weights are then updated using:

```text
weight = weight + learning_rate * error * input
```

The bias is updated using:

```text
bias = bias + learning_rate * error
```

The model repeatedly processes the training samples over multiple epochs and adjusts its parameters whenever it makes an incorrect prediction.

## Feature Scaling

Income and credit score have significantly different numerical ranges.

For example:

```text
Income       → 20 - 70
Credit Score → 400 - 750
```

Therefore, standardization is applied:

```text
X_scaled = (X - mean) / standard_deviation
```

This places the features on comparable scales before training the perceptron.

The same training mean and standard deviation are also used when predicting new loan applications.

## Model Flow

```text
Income + Credit Score
          |
          v
    Feature Scaling
          |
          v
   Weighted Sum
      XW + b
          |
          v
    Step Function
          |
          v
     Prediction
       /     \
      0       1
 Rejected   Approved
          |
          v
 Compare with Actual
          |
          v
   Calculate Error
          |
          v
Update Weights & Bias
```

## Decision Boundary

A perceptron is a linear classifier.

For two input features, its decision boundary is:

```text
w1*x1 + w2*x2 + b = 0
```

Rearranging:

```text
x2 = -(w1*x1 + b) / w2
```

The project uses Matplotlib to visualize this decision boundary and show how the perceptron separates approved and rejected loan applications.

## Project Structure

```text
03-perceptron-networks/
│
├── main.py
├── perceptron.py
└── README.md
```

### `perceptron.py`

Contains the custom `Perceptron` class responsible for:

* Weight initialization
* Step activation
* Model training
* Error calculation
* Weight updates
* Bias updates
* Prediction

### `main.py`

Responsible for:

* Creating the dataset
* Feature scaling
* Creating the perceptron model
* Training the model
* Making predictions
* Calculating accuracy
* Predicting a new loan application
* Visualizing the decision boundary

## Technologies Used

* Python
* NumPy
* Matplotlib

## Run the Project

Install the required libraries:

```bash
pip install numpy matplotlib
```

Run:

```bash
python main.py
```

The program displays the number of classification errors during each training epoch, final predictions, model accuracy, a prediction for a new loan application, and the learned decision boundary.

## Current Limitations

This project currently uses a small manually created dataset for understanding the perceptron algorithm.

A single perceptron can only learn **linearly separable classification problems**. Real-world loan approval decisions are more complex and may not be perfectly separable using a single linear decision boundary.

## Future Improvements

Future versions of this project can include:

* A larger dataset
* Train-test splitting
* Real-world loan approval data
* Additional loan application features
* Confusion matrix
* Precision, recall, and F1-score
* Comparison with Logistic Regression
* Comparison with a Multi-Layer Perceptron (MLP)

## Learning Outcome

This project demonstrates how a basic artificial neuron can learn a binary classification problem from scratch.

Instead of directly using a machine learning library's built-in Perceptron model, the core learning algorithm is implemented using NumPy to understand how weights, bias, activation functions, errors, and parameter updates work internally.
