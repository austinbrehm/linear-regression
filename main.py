# the goal of this program is to predict the price of coffee based on the size (oz)
# use linear regression with gradient descent to minimize cost function
import numpy as np
import matplotlib.pyplot as plt
import math

# data: x_train = size, y_train = price
size = np.array([8, 12, 16, 20])
price = np.array([1.50, 1.75, 2, 2.50])


def plot(x, y, title, x_label, y_label):
    plt.scatter(x, y, marker='x', c='r')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def compute_cost_function(x, y, w, b):
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        y_hat = w * x[i] + b
        cost = cost + (y_hat - y[i])**2
    cost = cost / (2 * m)
    return cost


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


def gradient_descent(x, y, w, b, cost_function, gradient_function, alpha, iterations):
    cost_history = []
    for i in range(iterations):
        dj_db, dj_dw = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            cost_history.append(cost_function(x, y, w, b))
        if i % math.ceil(iterations / 10) == 0:
            print(f'Iteration {i:4d}: Cost {cost_history[-1]:8.2f}')
    return w, b, cost_history


# initialize w, b, alpha (learning rate), iterations
initial_w = 1
initial_b = 1
initial_alpha = 0.0005
initial_iterations = 100

# plot initial data
# plot(size, price, 'Initial Data: Coffee Prices', 'size (oz)', 'price ($)')

# compute initial cost using initial parameters
initial_cost = compute_cost_function(size, price, initial_w, initial_b)
print(f'Initial Cost: {initial_cost}')

# compute gradient
w_gradient, b_gradient = compute_gradient(size, price, initial_w, initial_b)
print(f'dj_dw: {w_gradient}')
print(f'dj_db: {b_gradient}')

# compute w and b using gradient descent
w_final, b_final, cost_history_final = gradient_descent(size, price, initial_w, initial_b, compute_cost_function,
                                                        compute_gradient, initial_alpha, initial_iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
print(f'w_final: {w_final}')
print(f'b_final: {b_final}')
final_cost = compute_cost_function(size, price, w_final, b_final)
print(f'Final Cost: {final_cost}')

# predict price for 10 oz coffee using w_final and b_final
size_10 = w_final * 10 + b_final
print(f'Price prediction for 10 oz: ${size_10}')