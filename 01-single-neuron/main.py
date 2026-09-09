import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Training data
X = np.array([
    [1, 30],
    [2, 40],
    [3, 45],
    [4, 55],
    [5, 60],
    [6, 70],
    [7, 80],
    [8, 90]
], dtype=float)


y = np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1
], dtype=float)


# Normalize inputs
X[:, 0] = X[:, 0] / 8
X[:, 1] = X[:, 1] / 100


# Random weights
np.random.seed(42)

weights = np.random.randn(2)
bias = 0.0


learning_rate = 0.5
epochs = 5000


print("Initial weights:", weights)
print("Initial bias:", bias)


# -----------------------------
# Training
# -----------------------------

for epoch in range(epochs):

    # 1. Forward propagation
    z = np.dot(X, weights) + bias

    predictions = sigmoid(z)


    # 2. Calculate error
    error = predictions - y


    # 3. Calculate gradients
    dw = np.dot(X.T, error) / len(X)

    db = np.mean(error)


    # 4. Update weights and bias
    weights = weights - learning_rate * dw

    bias = bias - learning_rate * db


    # Print progress
    if epoch % 1000 == 0:

        loss = -np.mean(
            y * np.log(predictions + 1e-8)
            +
            (1 - y) * np.log(1 - predictions + 1e-8)
        )

        print(
            "Epoch:",
            epoch,
            "Loss:",
            round(loss, 4)
        )


print("\nLearned weights:", weights)
print("Learned bias:", bias)


# -----------------------------
# Predictions after training
# -----------------------------

final_predictions = sigmoid(
    np.dot(X, weights) + bias
)


print("\nPredictions after training:")

for i in range(len(X)):

    predicted_class = 1 if final_predictions[i] >= 0.5 else 0

    print(
        "Student:",
        X[i],
        "Actual:",
        int(y[i]),
        "Probability:",
        round(final_predictions[i], 3),
        "Prediction:",
        predicted_class
    )


# -----------------------------
# Predict a new student
# -----------------------------

study_hours = float(input("\nEnter study hours: "))
attendance = float(input("Enter attendance percentage: "))


# Normalize exactly like training data
study_hours = study_hours / 8
attendance = attendance / 100


new_student = np.array([
    study_hours,
    attendance
])


# Weighted sum
z = np.dot(new_student, weights) + bias


# Sigmoid activation
probability = sigmoid(z)


print("\nPass probability:", round(probability, 3))


if probability >= 0.5:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")