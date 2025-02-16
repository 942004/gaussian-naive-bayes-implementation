# -*- coding: utf-8 -*-
"""Untitled11.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12erf_6K_vL3wNdg7zvLXVeiGQWpG_gY2
"""

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


iris = datasets.load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class_stats = {}
for c in np.unique(y_train):
    X_c = X_train[y_train == c]
    class_stats[c] = {
        "mean": np.mean(X_c, axis=0),
        "std": np.std(X_c, axis=0)
    }

def gaussian_probability(x, mean, std):
    exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

def predict(X):
    predictions = []
    for x in X:
        posteriors = []
        for c, stats in class_stats.items():
            prior = np.log(len(y_train[y_train == c]) / len(y_train))
            likelihood = np.sum(np.log(gaussian_probability(x, stats["mean"], stats["std"])))
            posteriors.append(prior + likelihood)
        predictions.append(np.argmax(posteriors))
    return np.array(predictions)


y_pred = predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)
plt.figure(figsize=(8,6))
for c in np.unique(y_train):
    plt.scatter(X_pca[y_train == c, 0], X_pca[y_train == c, 1], label=iris.target_names[c])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.title("Iris Dataset PCA Visualization")
plt.show()