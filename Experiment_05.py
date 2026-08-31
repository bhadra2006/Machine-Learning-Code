# EXPERIMENT 05

# Title:
# Comparison of Maximum Likelihood Estimation and Maximum A Posteriori
# Estimation for Multinomial Text Data Using a Dirichlet Prior

# Aim:
# To implement Maximum Likelihood Estimation (MLE) and Maximum A Posteriori
# (MAP) estimation for multinomial text data and analyze the effect of
# different Dirichlet prior alpha values on word probability distributions.

# Algorithm:
# 1. Import the required libraries for data processing, numerical
#    computation, visualization, and text feature extraction.
# 2. Load selected categories from the 20 Newsgroups dataset.
# 3. Remove headers, footers, and quoted text from the documents.
# 4. Extract the documents and their corresponding class labels.
# 5. Convert the text documents into a numerical count matrix using
#    CountVectorizer.
# 6. Remove English stop words and limit the vocabulary to 5000 features.
# 7. Split the dataset into training data and testing data.
# 8. Estimate word probabilities for each class using Maximum Likelihood
#    Estimation (MLE).
# 9. Define multiple alpha values for the Dirichlet prior.
# 10. Estimate word probabilities using Maximum A Posteriori (MAP)
#     estimation for each alpha value.
# 11. Compare the word probabilities obtained using MLE and MAP.
# 12. Identify the most probable words for the selected class.
# 13. Analyze the effect of different Dirichlet prior alpha values on
#     the estimated word probabilities.
# 14. Visualize the relationship between alpha values and word
#     probabilities.
# 15. Plot and compare the probability estimates obtained using
#     MLE and MAP estimation.
# 16. Display the effect of different Dirichlet prior values on
#     probability distributions.

# Program:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


categories = [
    "sci.space",
    "rec.sport.baseball",
    "comp.graphics",
    "talk.politics.misc",
]


dataset = fetch_20newsgroups(
    subset="train",
    categories=categories,
    remove=("headers", "footers", "quotes")
)


documents, labels = dataset.data, dataset.target


print("Number of Documents:", len(documents))

print("Number of Classes:", len(categories))


vectorizer = CountVectorizer(
    stop_words="english",
    max_features=5000
)


X = vectorizer.fit_transform(documents).toarray()

vocabulary = vectorizer.get_feature_names_out()


print("Count Matrix Shape:", X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=42
)


def multinomial_MLE(X, y):

    classes = np.unique(y)

    probabilities = {}

    for c in classes:

        word_counts = np.sum(
            X[y == c],
            axis=0
        )

        probabilities[c] = (
            word_counts / np.sum(word_counts)
        )

    return probabilities


mle_prob = multinomial_MLE(
    X_train,
    y_train
)


print("\nMLE Estimation Completed")


def multinomial_MAP(X, y, alpha):

    classes = np.unique(y)

    probabilities = {}

    V = X.shape[1]

    for c in classes:

        word_counts = np.sum(
            X[y == c],
            axis=0
        )

        probabilities[c] = (
            (word_counts + alpha - 1)
            /
            (
                np.sum(word_counts)
                + V * (alpha - 1)
            )
        )

    return probabilities


alpha_values = [0.1, 0.5, 1, 5, 10]


map_results = {
    alpha: multinomial_MAP(
        X_train,
        y_train,
        alpha
    )
    for alpha in alpha_values
}


for alpha in alpha_values:

    print(
        "MAP completed for alpha =",
        alpha
    )


comparison = pd.DataFrame(
    {
        "Word": vocabulary,
        "MLE Probability": mle_prob[0],
        "MAP alpha=0.1": map_results[0.1][0],
        "MAP alpha=1": map_results[1][0],
        "MAP alpha=10": map_results[10][0],
    }
)


print(
    "\nProbability Comparison\n",
    comparison.head(20)
)


def top_words(probabilities, n=15):

    return vocabulary[
        np.argsort(probabilities)[-n:]
    ]


print(
    "\nTop Words using MLE\n",
    top_words(mle_prob[0])
)


print(
    "\nTop Words using MAP alpha=10\n",
    top_words(map_results[10][0])
)


prob_values = [
    map_results[alpha][0][10]
    for alpha in alpha_values
]


plt.figure(figsize=(8, 5))

plt.plot(
    alpha_values,
    prob_values,
    marker="o"
)

plt.xlabel("Dirichlet Prior Alpha")

plt.ylabel("Estimated Word Probability")

plt.title(
    "Effect of Dirichlet Prior on Multinomial Probability"
)

plt.grid()

plt.show()


plt.figure(figsize=(10, 5))

plt.plot(
    mle_prob[0][:50],
    label="MLE"
)


for alpha in [0.1, 1, 10]:

    plt.plot(
        map_results[alpha][0][:50],
        label=f"MAP alpha={alpha}"
    )


plt.xlabel("Vocabulary Index")

plt.ylabel("Probability")

plt.title(
    "MLE vs MAP Probability Estimates"
)

plt.legend()

plt.grid()

plt.show()


print(
    "\nEffect of Dirichlet Prior\n"
    "alpha < 1: Encourages sparse probability distributions; "
    "data dominates.\n"
    "alpha = 1: Equivalent to Laplace smoothing.\n"
    "alpha > 1: Smooths distribution toward uniform; "
    "reduces data dependence."
)
