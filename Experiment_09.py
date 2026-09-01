# EXPERIMENT 09

# Title:
# Customer Segmentation Using ID3 Decision Tree Classification

# Aim:
# To perform customer segmentation based on retail transaction data using
# the ID3 Decision Tree algorithm and analyze the importance of different
# customer-related features.

# Algorithm:
# 1. Import the required libraries for numerical computation, data
#    processing, visualization, model training, and evaluation.
# 2. Load the Online Retail dataset from an Excel file.
# 3. If the dataset is unavailable, generate a synthetic retail dataset.
# 4. Remove records with missing Customer ID values.
# 5. Convert Quantity, Unit Price, Customer ID, and Invoice Date into
#    appropriate data types.
# 6. Remove invalid or missing values from the dataset.
# 7. Remove transactions with zero or negative Quantity and Unit Price.
# 8. Create a Total Amount feature by multiplying Quantity and Unit Price.
# 9. Aggregate transaction data based on Customer ID.
# 10. Generate customer features such as Total Spending, Total Quantity,
#     Number of Invoices, Unique Products, Average Transaction Value,
#     and Recency.
# 11. Rank customers based on their Total Spending.
# 12. Divide customers into Low Value, Medium Value, and High Value
#     segments using quantile-based segmentation.
# 13. Select the required customer features as input variables.
# 14. Use the customer segment as the target variable.
# 15. Split the dataset into training data and testing data.
# 16. Train a Decision Tree Classifier using the entropy criterion,
#     representing the ID3 decision tree approach.
# 17. Predict customer segments using the trained model.
# 18. Calculate the classification accuracy of the model.
# 19. Calculate the importance of each input feature.
# 20. Display the ID3 decision tree.
# 21. Visualize the feature importance using a bar graph.

# Program:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


try:

    df = pd.read_excel(
        "Online Retail.xlsx"
    )

    print(
        "Dataset loaded successfully!"
    )


except FileNotFoundError:

    print(
        "Dataset not found. Using synthetic dataset..."
    )


    np.random.seed(42)

    n = 1000


    df = pd.DataFrame(
        {
            "InvoiceNo": np.random.randint(
                500000,
                505000,
                n
            ),

            "StockCode": np.random.randint(
                10000,
                90000,
                n
            ),

            "Quantity": np.random.randint(
                1,
                20,
                n
            ),

            "UnitPrice": np.random.uniform(
                1.0,
                50.0,
                n
            ).round(2),

            "InvoiceDate": pd.date_range(
                "2025-01-01",
                periods=n,
                freq="h"
            ),

            "CustomerID": np.random.randint(
                10000,
                10200,
                n
            )
        }
    )


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


df["CustomerID"] = df[
    "CustomerID"
].astype(int)


df["TotalAmount"] = (
    df["Quantity"]
    *
    df["UnitPrice"]
)


ref_date = (
    df["InvoiceDate"].max()
    +
    pd.Timedelta(days=1)
)


customer_data = df.groupby(
    "CustomerID"
).agg(

    TotalSpending=(
        "TotalAmount",
        "sum"
    ),

    TotalQuantity=(
        "Quantity",
        "sum"
    ),

    NumberOfInvoices=(
        "InvoiceNo",
        "nunique"
    ),

    UniqueProducts=(
        "StockCode",
        "nunique"
    ),

    AverageTransactionValue=(
        "TotalAmount",
        "mean"
    ),

    Recency=(
        "InvoiceDate",
        lambda x: (
            ref_date - x.max()
        ).days
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


X = customer_data[
    features
]


y = customer_data[
    "Segment"
]


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


print(
    "\n" + "=" * 45
)

print(
    "CUSTOMER SEGMENTATION USING ID3"
)

print(
    "=" * 45
)

print(
    f"Total Customers : {len(customer_data)}"
)

print(
    f"Model Accuracy  : {accuracy:.2%}"
)

print(
    "\nModel trained successfully!"
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

    fontweight="bold",

    pad=15
)


importance.plot(

    kind="bar",

    ax=axes[1]
)


axes[1].set_title(

    "Feature Importance Breakdown",

    fontsize=15,

    fontweight="bold",

    pad=15
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
