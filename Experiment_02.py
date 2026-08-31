# EXPERIMENT 02

# Title:
# Implementation and Comparison of Linear Regression and
# Polynomial Regression Using the Auto MPG Dataset

# Aim:
# To implement Linear Regression and Polynomial Regression
# models using the Auto MPG dataset and compare their performance
# using Mean Squared Error (MSE) and R² score.

# Algorithm:
# 1. Import the required libraries: NumPy, Pandas, Matplotlib,
#    and Scikit-learn modules.
# 2. Load the Auto MPG dataset from the UCI Machine Learning
#    Repository.
# 3. Assign appropriate column names to the dataset.
# 4. Remove missing values from the dataset.
# 5. Select Displacement as the input feature and MPG as the
#    target variable.
# 6. Split the dataset into training data (80%) and testing
#    data (20%).
# 7. Create and train a Linear Regression model using the
#    training data.
# 8. Predict the target values using the Linear Regression model.
# 9. Calculate the Mean Squared Error (MSE) and R² score for
#    the Linear Regression model.
# 10. Transform the input data into polynomial features of
#     degree 2.
# 11. Create and train a Polynomial Regression model.
# 12. Predict the target values using the Polynomial Regression
#     model.
# 13. Calculate the Mean Squared Error (MSE) and R² score for
#     the Polynomial Regression model.
# 14. Generate a smooth range of input values for visualization.
# 15. Plot the Linear Regression and Polynomial Regression
#     models along with the test data.
# 16. Display the performance metrics inside the graphs.
# 17. Display the complete dataset.

# Program:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# Load dataset

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

names = [
    "MPG", "Cylinders", "Displacement", "Horsepower",
    "Weight", "Acceleration", "Year", "Origin", "Name"
]

df = pd.read_csv(
    url,
    names=names,
    na_values="?",
    sep=r"\s+"
).dropna()


# Features and target

X = df[["Displacement"]]

y = df["MPG"]


# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Linear Regression

lin_model = LinearRegression()

lin_model.fit(X_train, y_train)

y_pred_lin = lin_model.predict(X_test)

lin_mse = mean_squared_error(
    y_test,
    y_pred_lin
)

lin_r2 = r2_score(
    y_test,
    y_pred_lin
)


# Polynomial Regression

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)

X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()

poly_model.fit(
    X_train_poly,
    y_train
)

y_pred_poly = poly_model.predict(
    X_test_poly
)

poly_mse = mean_squared_error(
    y_test,
    y_pred_poly
)

poly_r2 = r2_score(
    y_test,
    y_pred_poly
)


# Create Smooth Curve

X_line = np.linspace(
    X["Displacement"].min(),
    X["Displacement"].max(),
    300
).reshape(-1, 1)


# Plot Graphs

fig, (ax1, ax2) = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)


# Linear Regression Plot

ax1.scatter(
    X_test,
    y_test,
    color="gray",
    alpha=0.7,
    label="Test Data"
)

ax1.plot(
    X_line,
    lin_model.predict(X_line),
    color="red",
    linewidth=2,
    label="Linear Fit"
)

ax1.set_title("Linear Regression")

ax1.set_xlabel("Displacement")

ax1.set_ylabel("MPG")

ax1.legend()

ax1.grid(True)


# Display metrics inside graph

ax1.text(
    0.05,
    0.95,
    f"MSE = {lin_mse:.2f}\nR² = {lin_r2:.2f}",
    transform=ax1.transAxes,
    fontsize=11,
    verticalalignment="top",
    bbox=dict(
        facecolor="white",
        edgecolor="black",
        alpha=0.8
    )
)


# Polynomial Regression Plot

ax2.scatter(
    X_test,
    y_test,
    color="gray",
    alpha=0.7,
    label="Test Data"
)

ax2.plot(
    X_line,
    poly_model.predict(
        poly.transform(X_line)
    ),
    color="blue",
    linewidth=2,
    label="Polynomial Fit"
)

ax2.set_title(
    "Polynomial Regression (Degree 2)"
)

ax2.set_xlabel("Displacement")

ax2.set_ylabel("MPG")

ax2.legend()

ax2.grid(True)


# Display metrics inside graph

ax2.text(
    0.05,
    0.95,
    f"MSE = {poly_mse:.2f}\nR² = {poly_r2:.2f}",
    transform=ax2.transAxes,
    fontsize=11,
    verticalalignment="top",
    bbox=dict(
        facecolor="white",
        edgecolor="black",
        alpha=0.8
    )
)


plt.tight_layout()

plt.show()


# Display dataset

print("\n======================= DATASET =======================")

print(df.to_string(index=False))

print("=======================================================")
