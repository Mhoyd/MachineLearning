import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier

X1, y1 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=[[2.0, 2.0]],
                    cluster_std=0.75,
                    random_state=42)

X2, y2 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=[[3.0, 3.0]],
                    cluster_std=0.75,
                    random_state=42)

X = np.vstack((X1, X2))
y = np.hstack((np.zeros(len(X1)), np.ones(len(X2))))

clf = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', max_iter=1000, random_state=42)
clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
plt.scatter(X1[:, 0], X1[:, 1], c='red', label="Class 1", edgecolors='k')  # Class 1
plt.scatter(X2[:, 0], X2[:, 1], c='blue', label="Class 2", edgecolors='k')  # Class 2
plt.title("Decision plane")
plt.xlabel("Feature x1")
plt.ylabel("Feature x2")
plt.legend()
plt.grid(True)
plt.show()
