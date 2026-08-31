# EXPERIMENT 04

# Title:
# Comparison of Maximum Likelihood Estimation and Maximum A Posteriori
# Estimation Using Logistic Regression

# Aim:
# To implement and compare Logistic Regression models based on Maximum
# Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) estimation
# using L1 and L2 regularization on the Breast Cancer dataset.

# Algorithm:
# 1. Import the required libraries for data loading, preprocessing,
#    model training, evaluation, and visualization.
# 2. Load the Breast Cancer dataset.
# 3. Split the dataset into training data (80%) and testing data (20%)
#    while maintaining the class distribution.
# 4. Standardize the training and testing features using StandardScaler.
# 5. Create an unregularized Logistic Regression model representing
#    Maximum Likelihood Estimation (MLE).
# 6. Create a Logistic Regression model with L1 regularization
#    representing Maximum A Posteriori (MAP) estimation with a
#    Laplace prior.
# 7. Create a Logistic Regression model with L2 regularization
#    representing Maximum A Posteriori (MAP) estimation with a
#    Gaussian prior.
# 8. Train all three models using the training dataset.
# 9. Predict the class labels for the testing dataset.
# 10. Calculate the accuracy of each model.
# 11. Display the accuracy values of the three models.
# 12. Visualize the accuracy comparison using a bar graph.
# 13. Plot the feature weights of the MLE model.
# 14. Plot the feature weights of the MAP model with L1 regularization.
# 15. Plot the feature weights of the MAP model with L2 regularization.
# 16. Compare the effect of regularization on model weights and accuracy.

# Program:

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


X, y = load_breast_cancer(return_X_y=True)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


mle = LogisticRegression(
    penalty=None,
    max_iter=10000,
    random_state=42
)


map_l1 = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    random_state=42
)


map_l2 = LogisticRegression(
    penalty="l2",
    random_state=42
)


mle.fit(X_train, y_train)

map_l1.fit(X_train, y_train)

map_l2.fit(X_train, y_train)


acc_mle = accuracy_score(
    y_test,
    mle.predict(X_test)
)


acc_l1 = accuracy_score(
    y_test,
    map_l1.predict(X_test)
)


acc_l2 = accuracy_score(
    y_test,
    map_l2.predict(X_test)
)


print("\n===== MODEL ACCURACY =====")

print(f"Accuracy for MLE Model : {acc_mle:.4f}")

print(f"Accuracy for MAP (L1) Model : {acc_l1:.4f}")

print(f"Accuracy for MAP (L2) Model : {acc_l2:.4f}")


fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 8)
)


axes[0, 0].bar(
    ["MLE", "MAP (L1)", "MAP (L2)"],
    [acc_mle * 100, acc_l1 * 100, acc_l2 * 100],
    color=["red", "blue", "green"]
)

axes[0, 0].set_ylabel("Accuracy (%)")

axes[0, 0].set_ylim(80, 100)

axes[0, 0].set_title("1. Accuracy Comparison")

axes[0, 0].grid(
    axis="y",
    linestyle="--",
    alpha=0.7
)


axes[0, 1].plot(
    mle.coef_[0],
    color="red",
    marker="o"
)

axes[0, 1].axhline(
    0,
    color="black",
    linestyle="--"
)

axes[0, 1].set_title(
    "2. MLE Weights (Unregularized)"
)

axes[0, 1].set_xlabel(
    "Feature Index"
)

axes[0, 1].set_ylabel(
    "Weight Value"
)

axes[0, 1].grid(True)


axes[1, 0].plot(
    map_l1.coef_[0],
    color="blue",
    marker="o"
)

axes[1, 0].axhline(
    0,
    color="black",
    linestyle="--"
)

axes[1, 0].set_title(
    "3. MAP L1 Weights (Sparse / Laplace Prior)"
)

axes[1, 0].set_xlabel(
    "Feature Index"
)

axes[1, 0].set_ylabel(
    "Weight Value"
)

axes[1, 0].grid(True)


axes[1, 1].plot(
    map_l2.coef_[0],
    color="green",
    marker="o"
)

axes[1, 1].axhline(
    0,
    color="black",
    linestyle="--"
)

axes[1, 1].set_title(
    "4. MAP L2 Weights (Shrunken / Gaussian Prior)"
)

axes[1, 1].set_xlabel(
    "Feature Index"
)

axes[1, 1].set_ylabel(
    "Weight Value"
)

axes[1, 1].grid(True)


plt.tight_layout()

plt.show()
