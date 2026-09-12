import numpy as np
import matplotlib.pyplot as plt

from adaline import Adaline

# Create dataset

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


# ---------------------------------
# 2. Standardize features
# ---------------------------------

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_scaled = (X - mean) / std


# create adaline model

model = Adaline(
    learning_rate = 0.01,
    epochs = 50
)

# Train the model

model.fit(X_scaled, y)

# Make prediction

predictions = model.predict(X_scaled)

# Calculate accuracy

accuracy = np.mean(predictions == y)*100

# Print results

print("\nPredictions:")
print(predictions)

print("\nActual values:")
print(y)

print("\nAccuracy:")
print(f"{accuracy:.2f}%")

print("\nWeights")
print(model.weights)

print("\nBias:")
print(model.bias)

#plot loss curve

plt.plot(model.losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Adaline training loss")

plt.show()


# Plot decision boundary

for i in range(len(X_scaled)):

    if y[i] == 1:
        plt.scatter(X_scaled[i][0], X_scaled[i][1], marker="o")
    else:
        plt.scatter(X_scaled[i][0], X_scaled[i][1], marker="x")


x1_values = np.linspace(
    X_scaled[:, 0].min() - 1,
    X_scaled[:, 0].max() + 1,
    100
)

x2_values = -(
    model.weights[0] * x1_values + model.bias
) / model.weights[1]


plt.plot(x1_values, x2_values)

plt.xlabel("Scaled X1")
plt.ylabel("Scaled X2")
plt.title("ADALINE Decision Boundary")

plt.grid()
plt.show()