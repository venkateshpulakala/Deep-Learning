import numpy as np

class Perceptron:

    def __init__(self, learning = 0.1, epochs = 20):

        self.learning_rate = learning
        self.epochs = epochs

        self.weights = None
        self.bias = 0.0

    def step_function(self, z):
        return 1 if z >= 0 else 0


    def fit(self, X, y):

        # Initialize weights
        self.weights = np.zeros(X.shape[1])

        # Training
        for epoch in range(self.epochs):

            total_errors = 0

            for i in range(len(X)):

                z = np.dot(X[i],self.weights) + self.bias

                prediction = self.step_function(z)

                error = y[i] - prediction

                self.weights = (
                    self.weights
                    + self.learning_rate*error*X[i]
                )

                self.bias = self.bias + self.learning_rate*error

                if error != 0:
                    total_errors += 1

                print(
                f"Epoch: {epoch + 1}, "
                f"Errors: {total_errors}"
                )


    def predict(self, X):

        predictions = []

        for i in range(len(X)):
            z = np.dot(X[i], self.weights) + self.bias

            prediction = self.step_function(z)  

            predictions.append(prediction)

        return np.array(predictions) 
