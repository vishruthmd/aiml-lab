import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load Titanic dataset
titanic = fetch_openml("titanic", version=1, as_frame=True)
df = titanic.frame


# Select useful features + target
df = df[['age', 'fare', 'sex', 'pclass', 'survived']].dropna()

# Encode sex: male=1, female=0
df['sex'] = (df['sex'] == 'male').astype(int)

X = df[['age', 'fare', 'sex', 'pclass']].values
y = df['survived'].astype(int).values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=9
)

# Standardization
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Logistic Regression
model = LogisticRegression(C=0.1)
model.fit(X_train_std, y_train)

# Accuracy
print("Accuracy:", model.score(X_test_std, y_test))

# Plot using age (0) and fare (1)
x_min, x_max = X_train_std[:, 0].min() - 1, X_train_std[:, 0].max() + 1
y_min, y_max = X_train_std[:, 1].min() - 1, X_train_std[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.1),
    np.arange(y_min, y_max, 0.1)
)

# Fix sex=0 (female) and pclass=3 for visualization
Z = model.predict(
    np.c_[xx.ravel(), yy.ravel(),
          np.zeros(xx.ravel().shape),      # sex = female
          np.full(xx.ravel().shape, 3)]    # pclass = 3
)

Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train_std[:, 0], X_train_std[:, 1], c=y_train)
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Logistic Regression (Titanic with Sex & Pclass)")
plt.show()
