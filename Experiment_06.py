# EXPERIMENT 06

# Title:
# Comparison of Logistic Regression Performance With and Without
# Feature Scaling

# Aim:
# To implement Logistic Regression on the Diabetes dataset and compare
# the performance of the model with and without feature scaling using
# Accuracy, Precision, Recall, and F1 Score.

# Algorithm:
# 1. Import the required libraries for data handling, visualization,
#    data preprocessing, model training, and performance evaluation.
# 2. Load the Diabetes dataset using Pandas.
# 3. Separate the input features and target variable.
# 4. Split the dataset into training data (80%) and testing data (20%).
# 5. Train a Logistic Regression model using the original unscaled
#    training data.
# 6. Predict the output values using the unscaled test data.
# 7. Standardize the training and testing features using StandardScaler.
# 8. Train a Logistic Regression model using the scaled training data.
# 9. Predict the output values using the scaled test data.
# 10. Calculate Accuracy, Precision, Recall, and F1 Score for both models.
# 11. Display the performance metrics for the model without feature scaling.
# 12. Display the performance metrics for the model with feature scaling.
# 13. Create a bar graph to compare the performance of both models.
# 14. Save and display the generated visualization.

# Program:

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


data = pd.read_csv("diabetes.csv")


X = data.drop(
    "Outcome",
    axis=1
)

y = data["Outcome"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)

pred1 = model.predict(
    X_test
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


model.fit(
    X_train_scaled,
    y_train
)

pred2 = model.predict(
    X_test_scaled
)


metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]


result1 = [
    accuracy_score(y_test, pred1),
    precision_score(y_test, pred1),
    recall_score(y_test, pred1),
    f1_score(y_test, pred1)
]


result2 = [
    accuracy_score(y_test, pred2),
    precision_score(y_test, pred2),
    recall_score(y_test, pred2),
    f1_score(y_test, pred2)
]


print("========== WITHOUT FEATURE SCALING ==========")

for m, v in zip(metrics, result1):
    print(f"{m}: {v:.3f}")


print("\n========== WITH FEATURE SCALING ==========")

for m, v in zip(metrics, result2):
    print(f"{m}: {v:.3f}")


x = range(
    len(metrics)
)


plt.figure(
    figsize=(8, 5)
)


plt.bar(
    [i - 0.2 for i in x],
    result1,
    width=0.4,
    color="mediumseagreen",
    label="Without Scaling"
)


plt.bar(
    [i + 0.2 for i in x],
    result2,
    width=0.4,
    color="mediumpurple",
    label="With Scaling"
)


plt.xticks(
    x,
    metrics
)

plt.ylabel("Score")

plt.title(
    "Logistic Regression Performance Comparison"
)

plt.ylim(0, 1.1)

plt.legend()


plt.savefig(
    "output.png"
)


plt.show()
