import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

# Load dataset
file_path = "/Users/bmanjushareddy/Downloads/DCT_withoutduplicate 6 1 1.csv"
df = pd.read_csv(file_path)

# Separate features and target
X = df.drop(columns=["LABEL"]).values
y = df["LABEL"].values

# A1: Function to calculate entropy
def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

entropy_value = entropy(y)
print(f"Entropy: {entropy_value}")

# A2: Function to calculate Gini index
def gini_index(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return 1 - np.sum([p**2 for p in probabilities])

gini_value = gini_index(y)
print(f"Gini Index: {gini_value}")

# A3: Function to determine the best feature to split using Information Gain
def best_feature_to_split(X, y):
    base_entropy = entropy(y)
    info_gains = []
    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])
        weighted_entropy = sum([(np.sum(X[:, feature] == v) / len(y)) * entropy(y[X[:, feature] == v]) for v in values])
        info_gains.append(base_entropy - weighted_entropy)
    return np.argmax(info_gains)

best_feature = best_feature_to_split(X, y)
print(f"Best feature to split: {best_feature}")

# A4: Function for binning continuous features
def bin_data(X, n_bins=4, strategy='uniform'):
    binner = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
    return binner.fit_transform(X.reshape(-1, 1)).astype(int)

# Example binning on first feature
binned_feature = bin_data(X[:, 0])
print(f"Binned Feature Sample: {binned_feature[:10].flatten()}")
