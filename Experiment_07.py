# EXPERIMENT 07

# Title:
# Comparison of Multinomial Naive Bayes and Bernoulli Naive Bayes
# for Text Classification

# Aim:
# To implement Multinomial Naive Bayes and Bernoulli Naive Bayes models
# for text classification using the 20 Newsgroups dataset and compare
# their performance using Accuracy and F1 Score.

# Algorithm:
# 1. Import the required libraries for dataset loading, text feature
#    extraction, model training, evaluation, and visualization.
# 2. Select the required categories from the 20 Newsgroups dataset.
# 3. Load the text dataset and extract the documents and target labels.
# 4. Split the dataset into training data (80%) and testing data (20%).
# 5. Convert the text documents into numerical feature vectors using
#    CountVectorizer.
# 6. Train a Multinomial Naive Bayes model using the training data.
# 7. Predict the class labels for the testing data using the
#    Multinomial Naive Bayes model.
# 8. Calculate the Accuracy and weighted F1 Score of the
#    Multinomial Naive Bayes model.
# 9. Train a Bernoulli Naive Bayes model using the training data.
# 10. Predict the class labels for the testing data using the
#     Bernoulli Naive Bayes model.
# 11. Calculate the Accuracy and weighted F1 Score of the
#     Bernoulli Naive Bayes model.
# 12. Compare the performance of both models.
# 13. Visualize the Accuracy and F1 Score using a bar graph.
# 14. Save and display the comparison graph.

# Program:

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, f1_score


categories = [
    "sci.space",
    "rec.sport.baseball",
    "comp.graphics",
    "talk.politics.misc",
]


data = fetch_20newsgroups(
    categories=categories
)

X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(
    X_train
)

X_test_vec = vectorizer.transform(
    X_test
)


mnb = MultinomialNB()

mnb.fit(
    X_train_vec,
    y_train
)

pred_mnb = mnb.predict(
    X_test_vec
)


acc_mnb = accuracy_score(
    y_test,
    pred_mnb
)

f1_mnb = f1_score(
    y_test,
    pred_mnb,
    average="weighted"
)


print("----- Multinomial Naive Bayes -----")

print("Accuracy :", acc_mnb)

print("F1 Score :", f1_mnb)


bnb = BernoulliNB()

bnb.fit(
    X_train_vec,
    y_train
)

pred_bnb = bnb.predict(
    X_test_vec
)


acc_bnb = accuracy_score(
    y_test,
    pred_bnb
)

f1_bnb = f1_score(
    y_test,
    pred_bnb,
    average="weighted"
)


print("\n----- Bernoulli Naive Bayes -----")

print("Accuracy :", acc_bnb)

print("F1 Score :", f1_bnb)


labels = [
    "MNB Acc",
    "BNB Acc",
    "MNB F1",
    "BNB F1",
]


scores = [
    acc_mnb,
    acc_bnb,
    f1_mnb,
    f1_bnb,
]


colors = [
    "blue",
    "green",
    "blue",
    "green"
]


plt.figure(
    figsize=(7, 5)
)


plt.bar(
    labels,
    scores,
    color=colors
)


plt.ylabel("Score")

plt.title(
    "Naive Bayes Model Comparison"
)

plt.ylim(0, 1)


plt.savefig(
    "naive_bayes_comparison.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()
