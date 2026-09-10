import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -----------------------------
# 1. XOR Dataset
# -----------------------------

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])


# -----------------------------
# 2. Initialize parameters
# -----------------------------

np.random.seed(42)

# Input layer → Hidden layer
W1 = np.random.randn(2, 3)
b1 = np.zeros((1, 3))

# Hidden layer → Output layer
W2 = np.random.randn(3, 1)
b2 = np.zeros((1, 1))


loss_history = []


epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):

    # -------------------------
    # 1. Forward Propagation
    # -------------------------

    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    predictions = sigmoid(Z2)


    # -------------------------
    # 2. Calculate Loss
    # -------------------------

    epsilon = 1e-8

    loss = -np.mean(
        y * np.log(predictions + epsilon)
        + (1 - y) * np.log(1 - predictions + epsilon)
    )

    loss_history.append(loss)


    # -------------------------
    # 3. Backpropagation
    # -------------------------

    m = len(X)

    # Output layer
    dZ2 = predictions - y

    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    # Hidden layer
    dA1 = np.dot(dZ2, W2.T)

    dZ1 = dA1 * A1 * (1 - A1)

    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.mean(dZ1, axis=0, keepdims=True)


    # -------------------------
    # 4. Update Parameters
    # -------------------------

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2


    # -------------------------
    # 5. Print Loss
    # -------------------------

    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.4f}")

# Final forward propagation

Z1 = np.dot(X, W1) + b1
A1 = sigmoid(Z1)

Z2 = np.dot(A1, W2) + b2
predictions = sigmoid(Z2)

final_predictions = (predictions >= 0.5).astype(int)

accuracy = np.mean(final_predictions == y) * 100

print("\nProbabilities:")
print(predictions)

print("\nPredictions:")
print(final_predictions)

print("\nActual values:")
print(y)

print(f"\nAccuracy: {accuracy:.2f}%")


# -------------------------
# Plot Loss Curve
# -------------------------

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Loss Curve of Multi-Layer Feed-Forward Network")
plt.savefig("results/loss_curve.png")
plt.show()