# EXPERIMENT 08

# Title:
# Performance Analysis of K-Nearest Neighbors Classification
# Using Different K Values on the Fashion-MNIST Dataset

# Aim:
# To implement the K-Nearest Neighbors (KNN) classification algorithm
# on the Fashion-MNIST dataset and analyze the effect of different
# K values on classification accuracy and prediction time.

# Algorithm:
# 1. Import the required libraries for time measurement,
#    visualization, dataset loading, model training, and evaluation.
# 2. Load the Fashion-MNIST dataset using OpenML.
# 3. Normalize the pixel values by dividing them by 255.
# 4. Convert the target labels into integer values.
# 5. Select 10,000 samples for training and 2,000 samples for testing.
# 6. Define different K values for the K-Nearest Neighbors classifier.
# 7. Create and train a KNN model for each K value.
# 8. Record the prediction time for each trained model.
# 9. Predict the class labels for the test dataset.
# 10. Calculate the classification accuracy for each K value.
# 11. Store the accuracy and prediction time values.
# 12. Display the K value, accuracy, and prediction time.
# 13. Plot a graph showing Accuracy versus K.
# 14. Plot a graph showing Prediction Time versus K.
# 15. Compare the performance of the KNN classifier for different
#     values of K.

# Program:

import time

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


mnist = fetch_openml(
    "Fashion-MNIST",
    version=1,
    as_frame=False
)


X, y = mnist.data / 255.0, mnist.target.astype(int)


X_train, X_test = X[:10000], X[60000:62000]

y_train, y_test = y[:10000], y[60000:62000]


K = [1, 3, 5, 7, 9]

accuracy, times = [], []


for k in K:

    model = KNeighborsClassifier(
        n_neighbors=k
    )

    model.fit(
        X_train,
        y_train
    )

    start = time.time()

    pred = model.predict(
        X_test
    )

    times.append(
        time.time() - start
    )

    accuracy.append(
        accuracy_score(
            y_test,
            pred
        ) * 100
    )


print("\nK Value | Accuracy | Time")


for k, a, t in zip(
    K,
    accuracy,
    times
):

    print(
        f"{k:^7} | {a:.2f}%   | {t:.3f}s"
    )


plt.figure(
    figsize=(10, 4)
)


plt.subplot(
    1,
    2,
    1
)


plt.plot(
    K,
    accuracy,
    "bo-"
)


plt.xlabel("K")

plt.ylabel("Accuracy (%)")

plt.title("Accuracy vs K")

plt.grid()


plt.subplot(
    1,
    2,
    2
)


plt.plot(
    K,
    times,
    "rs-"
)


plt.xlabel("K")

plt.ylabel("Time (s)")

plt.title("Time vs K")

plt.grid()


plt.tight_layout()

plt.show()
