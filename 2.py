import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

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

model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(X, y, epochs=250, batch_size=20, verbose=1)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = (Z > 0.5).astype(int).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5], colors=['r', 'b'], alpha=0.7)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', edgecolors="k", label="Class 1")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', edgecolors="k", label="Class 2")

plt.title("Decision Boundary - Neural Network")
plt.xlabel("Feature X1")
plt.ylabel("Feature X2")
plt.legend()
plt.show()

