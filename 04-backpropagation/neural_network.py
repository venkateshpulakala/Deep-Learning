import numpy as np


class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):

        self.learning_rate = learning_rate

        # Input -> Hidden weights
        self.W1 = np.random.randn(input_size, hidden_size)

        # Hidden bias
        self.b1 = np.zeros((1, hidden_size))

        # Hidden -> Output weights
        self.W2 = np.random.randn(hidden_size, output_size)

        # Output bias
        self.b2 = np.zeros((1, output_size))


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def sigmoid_derivative(self, x):
        return x * (1 - x)


    def forward(self, X):

        # Input -> Hidden
        self.z1 = np.dot(X, self.W1) + self.b1

        self.a1 = self.sigmoid(self.z1)

        # Hidden -> Output
        self.z2 = np.dot(self.a1, self.W2) + self.b2

        self.output = self.sigmoid(self.z2)

        return self.output


    def backward(self, X, y):

        # Output error
        output_error = self.output - y

        # Output delta
        output_delta = (
            output_error *
            self.sigmoid_derivative(self.output)
        )

        # Gradient for W2
        dW2 = np.dot(self.a1.T, output_delta)

        # Gradient for b2
        db2 = np.sum(output_delta, axis=0, keepdims=True)

        # Hidden layer error
        hidden_error = np.dot(output_delta, self.W2.T)

        # Hidden delta
        hidden_delta = (
            hidden_error *
            self.sigmoid_derivative(self.a1)
        )

        # Gradient for W1
        dW1 = np.dot(X.T, hidden_delta)

        # Gradient for b1
        db1 = np.sum(hidden_delta, axis=0, keepdims=True)

        # Update weights
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1


    def train(self, X, y, epochs):

        self.losses = []

        for epoch in range(epochs):

            # Forward propagation
            predictions = self.forward(X)

            # Calculate loss
            loss = np.mean((y - predictions) ** 2)

            # Store loss
            self.losses.append(loss)

            # Backpropagation
            self.backward(X, y)

            if epoch % 1000 == 0:
                print(f"Epoch: {epoch}, Loss: {loss:.6f}")


    def predict(self, X):

        predictions = self.forward(X)

        return (predictions >= 0.5).astype(int)