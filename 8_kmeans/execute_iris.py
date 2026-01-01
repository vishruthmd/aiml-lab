import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load Iris dataset
iris = load_iris()
X = iris.data

# Apply K-Means
k = 3
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Plot using first two features
colors = ['red', 'green', 'blue']

for i in range(k):
    plt.scatter(
        X[labels == i, 0],   # Sepal length
        X[labels == i, 1],   # Sepal width
        c=colors[i],
        label=f'Cluster {i}'
    )

# Plot centroids
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c='black',
    marker='x',
    s=100,
    label='Centroids'
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering (sklearn)")
plt.legend()
plt.show()
