from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = []
        for x_train in self.X_train:
            distances.append(np.linalg.norm(x - x_train)) 
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Load dataset from CSV
# Expected CSV format: columns are features, last column is target (class labels)
df = pd.read_csv('dataset.csv')

# Separate features and target
X = df.iloc[:, :-1].values  # All columns except last
y = df.iloc[:, -1].values   # Last column as target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create a k-NN classifier with 3 neighbors
knn = KNN(k=3)

# Train the model using the training sets
knn.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = knn.predict(X_test)
print('Accuracy: %.4f' % np.mean(y_pred == y_test))
print("Predictions:", y_pred)
