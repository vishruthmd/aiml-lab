import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

# Load Titanic dataset
titanic = fetch_openml("titanic", version=1, as_frame=True)
df = titanic.frame

# Select numeric features
features = ['age', 'fare', 'pclass', 'sibsp', 'parch']

df = df[features].dropna()

X = df.values
print(X)
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
k = 2
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

# Plot using actual features: age vs fare
colors = ['red', 'blue']

for i in range(k):
    plt.scatter(
        X_scaled[labels == i, 0],  # age
        X_scaled[labels == i, 1],  # fare
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

plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("K-Means Clustering (Titanic Dataset)")
plt.legend()
plt.show()


# the clustering of this program because titanic dataset has multiple features and is not suitable for 2D.
# but plotting only age and fare to show the output 
# fare feature is highly skewed
# kmeans also uses class numbers to calculate the distances but is kinda irrelevant for clustering

