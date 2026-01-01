import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_breast_cancer()
X = data.data
feature_names = data.feature_names

print("Feature names:", feature_names)
print("target names:", data.target_names)
print("data shape:", X.shape)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
k = 2
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

# Plot using first two actual features
colors = ['red', 'green']

for i in range(k):
    plt.scatter(
        X_scaled[labels == i, 0],  # mean radius, change 0 to some other number for other features
        X_scaled[labels == i, 1],  # mean texture, change 1 to some other number for other features
        c=colors[i],
        label=f'Cluster {i}'
    )

# Plot centroids
plt.scatter(
    centroids[:, 0], # chnge 0 to some other number for other features
    centroids[:, 1], # change 1 to some other number for other features
    c='black',
    marker='x',
    s=100,
    label='Centroids'
)

plt.xlabel(feature_names[0])  # mean radius
plt.ylabel(feature_names[1])  # mean texture
plt.title("K-Means Clustering (Breast Cancer Dataset)")
plt.legend()
plt.show()
