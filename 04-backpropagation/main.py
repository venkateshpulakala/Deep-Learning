import numpy as np
import matplotlib.pyplot as plt

from neural_network import NeuralNetwork


# XOR dataset
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


# Create neural network
model = NeuralNetwork(
    input_size=2,
    hidden_size=2,
    output_size=1,
    learning_rate=0.5
)


# Train model
model.train(
    X,
    y,
    epochs=10000
)


print("\nPredictions:")

predictions = model.predict(X)

for inputs, prediction in zip(X, predictions):

    print(
        f"Input: {inputs} -> Prediction: {prediction[0]}"
    )


print("\nLearned Parameters:")

print("\nInput -> Hidden Weights (W1):")
print(model.W1)

print("\nHidden Bias (b1):")
print(model.b1)

print("\nHidden -> Output Weights (W2):")
print(model.W2)

print("\nOutput Bias (b2):")
print(model.b2)

# Plot loss curve
plt.plot(model.losses)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Backpropagation Training Loss")

plt.show()