import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def kmeans(X, k):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(100):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

    return centroids, labels

# Load dataset from CSV
# Expected CSV format: columns are features (no target column for clustering)
df = pd.read_csv('dataset.csv')
X = df.values

# Apply custom k-means clustering
k = 3
centroids, labels = kmeans(X, k)

print(f"Cluster labels: {labels}")
print(f"Centroids:\n{centroids}")

# Plot the results (for 2D data only)
if X.shape[1] == 2:
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    
    for i in range(k):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], 
                   c=colors[i % len(colors)], label=f'Cluster {i+1}')
    
    plt.scatter(centroids[:, 0], centroids[:, 1], 
               marker='x', c='black', s=200, linewidths=3, label='Centroids')
    
    plt.title('K-Means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
