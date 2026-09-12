import numpy as np

class Adaline:

    def __init__(self, learning_rate = 0.01, epochs = 50):

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.weights = None
        self.bias = 0

        self.losses = []


    def fit(self, X, y):

        #Number of features
        number_of_features = X.shape[1]

        #Initialize weights with zeros
        self.weights = np.zeros(number_of_features)

        self.bias = 0

        for epoch in range(self.epochs):

            z = np.dot(X, self.weights) + self.bias

            errors = y - z

            loss = np.mean(errors**2)/2

            self.losses.append(loss)

            dw = -np.dot(X.T, errors)/len(X)

            db = -np.mean(errors)

            self.weights = self.weights - self.learning_rate*dw

            self.bias = self.bias - self.learning_rate * db


            print(
                f"Epoch: {epoch + 1}, " f"Loss: {loss:.4f}"
            )

    def predict(self, X):

        #Calculate linear output
        linear_output = np.dot(X, self.weights) + self.bias

        #Apply threshold
        predictions = []

        for value in linear_output:
            if value >= 0:
                predictions.append(1)
            else:
                predictions.append(-1)

        predictions = np.array(predictions)

        return predictions