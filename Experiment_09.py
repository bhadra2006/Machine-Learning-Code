# EXPERIMENT 09

# Title:
# Customer Segmentation Using ID3 Decision Tree Classification

# Aim:
# To perform customer segmentation using the ID3 Decision Tree algorithm
# based on customer retail transaction data and analyze the importance
# of different customer-related features.

# Algorithm:
# 1. Import the required libraries.
# 2. Load the Online Retail dataset from an Excel file.
# 3. Generate a synthetic dataset if the original dataset is unavailable.
# 4. Clean and preprocess the dataset by handling missing and invalid values.
# 5. Create a TotalAmount feature from Quantity and UnitPrice.
# 6. Aggregate transaction data to generate customer-level features.
# 7. Segment customers into Low, Medium, and High Value groups based on
#    their total spending.
# 8. Select customer-related features and define the target variable.
# 9. Split the dataset into training and testing sets.
# 10. Train a Decision Tree Classifier using the entropy criterion.
# 11. Predict customer segments and calculate model accuracy.
# 12. Calculate feature importance scores.
# 13. Visualize the decision tree and feature importance graph.

# Program:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


try:
    df = pd.read_excel("Online Retail.xlsx")
    print("Dataset loaded successfully!")

except FileNotFoundError:
    print("Dataset not found. Using synthetic dataset...")

    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "InvoiceNo": np.random.randint(500000, 505000, n),
        "StockCode": np.random.randint(10000, 90000, n),
        "Quantity": np.random.randint(1, 20, n),
        "UnitPrice": np.random.uniform(1.0, 50.0, n).round(2),
        "InvoiceDate": pd.date_range(
            "2025-01-01",
            periods=n,
            freq="h"
        ),
        "CustomerID": np.random.randint(10000, 10200, n)
    })


df = df.dropna(
    subset=["CustomerID"]
).copy()


df["Quantity"] = pd.to_numeric(
    df["Quantity"],
    errors="coerce"
)

df["UnitPrice"] = pd.to_numeric(
    df["UnitPrice"],
    errors="coerce"
)

df["CustomerID"] = pd.to_numeric(
    df["CustomerID"],
    errors="coerce"
)

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    errors="coerce"
)


df = df.dropna(
    subset=[
        "Quantity",
        "UnitPrice",
        "CustomerID",
        "InvoiceDate"
    ]
)


df = df[
    (df["Quantity"] > 0)
    &
    (df["UnitPrice"] > 0)
].copy()


df["CustomerID"] = df["CustomerID"].astype(int)


df["TotalAmount"] = (
    df["Quantity"] * df["UnitPrice"]
)


ref_date = (
    df["InvoiceDate"].max()
    + pd.Timedelta(days=1)
)


customer_data = df.groupby(
    "CustomerID"
).agg(
    TotalSpending=("TotalAmount", "sum"),
    TotalQuantity=("Quantity", "sum"),
    NumberOfInvoices=("InvoiceNo", "nunique"),
    UniqueProducts=("StockCode", "nunique"),
    AverageTransactionValue=("TotalAmount", "mean"),
    Recency=(
        "InvoiceDate",
        lambda x: (ref_date - x.max()).days
    )
).reset_index()


customer_data["SpendingRank"] = (
    customer_data["TotalSpending"]
    .rank(method="first")
)


customer_data["Segment"] = pd.qcut(
    customer_data["SpendingRank"],
    q=3,
    labels=[
        "Low Value",
        "Medium Value",
        "High Value"
    ]
)


customer_data.drop(
    columns=["SpendingRank"],
    inplace=True
)


features = [
    "TotalQuantity",
    "NumberOfInvoices",
    "UniqueProducts",
    "AverageTransactionValue",
    "Recency"
]


X = customer_data[features]

y = customer_data["Segment"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


dt = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=2,
    min_samples_leaf=8,
    random_state=42
)


dt.fit(
    X_train,
    y_train
)


y_pred = dt.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\nCUSTOMER SEGMENTATION USING ID3")

print(
    f"Total Customers : {len(customer_data)}"
)

print(
    f"Model Accuracy  : {accuracy:.2%}"
)

print(
    "Model trained successfully!"
)


importance = pd.Series(
    dt.feature_importances_,
    index=features
).sort_values(
    ascending=False
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(18, 7)
)


plot_tree(
    dt,
    feature_names=features,
    class_names=[
        str(x)
        for x in dt.classes_
    ],
    filled=True,
    rounded=True,
    fontsize=9,
    precision=2,
    ax=axes[0]
)


axes[0].set_title(
    "ID3 Decision Tree",
    fontsize=15,
    fontweight="bold"
)


importance.plot(
    kind="bar",
    ax=axes[1]
)


axes[1].set_title(
    "Feature Importance Breakdown",
    fontsize=15,
    fontweight="bold"
)

axes[1].set_xlabel(
    "Features"
)

axes[1].set_ylabel(
    "Importance Score"
)

axes[1].tick_params(
    axis="x",
    rotation=35
)


for container in axes[1].containers:
    axes[1].bar_label(
        container,
        fmt="%.2f",
        padding=3
    )


plt.tight_layout()

plt.show()
