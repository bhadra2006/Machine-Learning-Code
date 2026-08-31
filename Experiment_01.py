# EXPERIMENT 01

# Title:
# Implementation and Comparison of Simple Linear Regression using
# Gradient Descent and Normal Equation

# Aim:
# To implement Simple Linear Regression using Gradient Descent and
# the Normal Equation and compare their performance using the
# California Housing dataset.

# Algorithm:
# 1. Import the required libraries.
# 2. Load the California Housing dataset.
# 3. Select Median Income (MedInc) as the independent variable.
# 4. Filter the dataset by removing values greater than or equal to 15.
# 5. Randomly shuffle the dataset.
# 6. Split the dataset into training data (80%) and testing data (20%).
# 7. Calculate the mean and standard deviation of the training data.
# 8. Standardize the training feature.
# 9. Initialize the parameters for Gradient Descent.
# 10. Train the Simple Linear Regression model using Gradient Descent.
# 11. Convert the Gradient Descent parameters back to the original scale.
# 12. Calculate the regression parameters using the Normal Equation.
# 13. Predict the output values for the test dataset.
# 14. Calculate Mean Squared Error (MSE) and R² score.
# 15. Visualize and compare the regression lines obtained using both methods.

# Program:

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)

X = X[:, 0]

mask = X < 15

X, y = X[mask], y[mask]

np.random.seed(42)

idx = np.random.permutation(len(y))

split = int(len(y) * 0.8)

X_tr, X_te = X[idx[:split]], X[idx[split:]]
y_tr, y_te = y[idx[:split]], y[idx[split:]]

mu = np.mean(X_tr)

sigma = np.std(X_tr)

X_tr_s = (X_tr - mu) / sigma

t0, t1 = 0.0, 0.0

learning_rate = 0.01

iterations = 3000

for _ in range(iterations):

    y_pred = t0 + t1 * X_tr_s

    err = y_pred - y_tr

    t0 -= learning_rate * (2 / len(y_tr)) * np.sum(err)

    t1 -= learning_rate * (2 / len(y_tr)) * np.sum(err * X_tr_s)

slope_gd = t1 / sigma

int_gd = t0 - (t1 * mu / sigma)


X_mat = np.column_stack((np.ones(len(X_tr)), X_tr))

theta_ne = np.linalg.inv(X_mat.T @ X_mat) @ X_mat.T @ y_tr

int_ne = theta_ne[0]

slope_ne = theta_ne[1]


plt.style.use("seaborn-v0_8-whitegrid")

x_line = np.array([X_te.min(), X_te.max()])


def make_plot(title, intercept, slope, color, linestyle="-"):

    y_pred = intercept + slope * X_te

    mse = np.mean((y_te - y_pred) ** 2)

    r2 = 1 - (
        np.sum((y_te - y_pred) ** 2)
        / np.sum((y_te - np.mean(y_te)) ** 2)
    )

    plt.figure(figsize=(7, 4.5))

    plt.scatter(
        X_te,
        y_te,
        color="royalblue",
        alpha=0.4,
        edgecolors="white",
        label="Test Data",
    )

    plt.plot(
        x_line,
        intercept + slope * x_line,
        color=color,
        linewidth=3,
        linestyle=linestyle,
        label="Regression Line",
    )

    plt.title(
        f"{title}\nMSE = {mse:.4f} | R² = {r2:.4f}",
        fontweight="bold",
    )

    plt.xlabel("Median Income (MedInc)")

    plt.ylabel("Median House Value")

    plt.legend()

    plt.grid(True, linestyle="--", alpha=0.5)


make_plot(
    "Method A: Gradient Descent",
    int_gd,
    slope_gd,
    "crimson",
)

make_plot(
    "Method B: Normal Equation",
    int_ne,
    slope_ne,
    "darkorange",
    "--",
)

plt.show()
