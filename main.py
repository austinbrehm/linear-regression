import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([952, 1244, 1947])
y_train = np.array([271.5, 232, 509.8])


def plot(x, y, title, x_label, y_label):
    plt.scatter(x, y, marker='x', c='r')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


#plot(x_train, y_train, 'title', 'x-label', 'y-label')


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(0, m):
        y_hat = w * x[i] + b
        dj_dw_1 = (y_hat - y[i]) * x[i]
        dj_db_1 = y_hat - y[i]
        dj_dw += dj_dw_1
        dj_db += dj_db_1

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


dj_dw, dj_db = compute_gradient(x_train, y_train, 100, 100)

print(f'dj_dw: {dj_dw}')
print(f'dj_db: {dj_db}')