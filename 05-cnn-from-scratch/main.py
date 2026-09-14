import numpy as np

from convolution import convolution
from pooling import max_pooling
from functions import softmax, cross_entropy


# -----------------------
# Input image
# -----------------------

image = np.array([
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0]
])


# -----------------------
# Fixed convolution kernel
# -----------------------

kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])


# -----------------------
# CNN feature extraction
# -----------------------

feature_map = convolution(image, kernel)

relu_output = np.maximum(0, feature_map)

pooled_output = max_pooling(
    relu_output,
    pool_size=2,
    stride=1
)

x = pooled_output.flatten()


print("Flattened Features:")
print(x)


# -----------------------
# Dense layer
# -----------------------

np.random.seed(42)

weights = np.random.randn(4, 2) * 0.1

bias = np.zeros(2)


# 0 = Vertical
# 1 = Horizontal

correct_class = 0


learning_rate = 0.1
epochs = 20


for epoch in range(epochs):

    # -----------------------
    # Forward pass
    # -----------------------

    scores = np.dot(x, weights) + bias

    probabilities = softmax(scores)

    loss = cross_entropy(
        probabilities,
        correct_class
    )


    # -----------------------
    # Create one-hot target
    # -----------------------

    target = np.zeros(2)

    target[correct_class] = 1


    # -----------------------
    # Gradient of loss
    # -----------------------

    dscores = probabilities - target


    # -----------------------
    # Gradients for weights
    # -----------------------

    dweights = np.outer(x, dscores)

    dbias = dscores


    # -----------------------
    # Gradient descent
    # -----------------------

    weights = weights - learning_rate * dweights

    bias = bias - learning_rate * dbias


    print(
        f"Epoch {epoch + 1}, "
        f"Loss: {loss:.4f}, "
        f"Probabilities: {probabilities}"
    )