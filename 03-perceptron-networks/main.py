import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron

# -------------------------------------------------
# 1. Dataset
# -------------------------------------------------

X = np.array([
    [20, 400],
    [25, 450],
    [30, 500],
    [35, 520],
    [40, 580],
    [50, 650],
    [60, 700],
    [70, 750]
], dtype=float)

y = np.array([
    0, 0, 0, 0,
    1, 1, 1, 1
])

# -------------------------------------------------
# 2. Feature Scaling
# -------------------------------------------------

mean = X.mean(axis=0)
std = X.std(axis=0)

X_scaled = (X - mean) / std


# -------------------------------------------------
# 3. Create Perceptron Model
# -------------------------------------------------

model = Perceptron(
    learning=0.1,
    epochs=20
)


# -------------------------------------------------
# 4. Train Model
# -------------------------------------------------

model.fit(X_scaled, y)

# -------------------------------------------------
# 5. Predictions
# -------------------------------------------------

predictions = model.predict(X_scaled)


# -------------------------------------------------
# 6. Accuracy
# -------------------------------------------------

accuracy = np.mean(predictions == y) * 100

print(f"Accuracy: {accuracy:.2f}%")


# -------------------------------------------------
# 7. New Loan Prediction
# -------------------------------------------------

new_loan = np.array([
    [45, 620]
], dtype=float)

new_loan_scaled = (new_loan - mean) / std

prediction = model.predict(new_loan_scaled)

print("\nNew Loan")

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")




# -------------------------------------------------
# 8. Decision Boundary
# -------------------------------------------------

for i in range(len(X_scaled)):

    if y[i] == 0:

        plt.scatter(
            X_scaled[i][0],
            X_scaled[i][1],
            marker="o"
        )

    else:

        plt.scatter(
            X_scaled[i][0],
            X_scaled[i][1],
            marker="x"
        )


x1_values = np.linspace(
    X_scaled[:, 0].min() - 1,
    X_scaled[:, 0].max() + 1,
    100
)

x2_values = -(
    model.weights[0] * x1_values
    + model.bias
) / model.weights[1]

plt.plot(
    x1_values,
    x2_values
)

plt.xlabel("Income (Scaled)")
plt.ylabel("Credit Score (Scaled)")

plt.title(
    "Loan Approval Using Perceptron"
)

plt.show()