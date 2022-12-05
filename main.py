# the goal of this program is to predict the price of coffee based on the size (oz)
# use linear regression with stochastic gradient descent to find w,b that minimizes cost function
import numpy as np
import matplotlib.pyplot as plt
import math

# data: x_train = size, y_train = price
size = np.array([8, 12, 16, 20, 8, 8, 8, 8, 8, 12, 12, 12, 12, 12])
price = np.array([1.50, 1.75, 2, 2.50, 1.55, 1.60, 1.45, 1.60, 1.58, 1.80, 1.70, 1.75, 1.72, 1.78])


def plot(x, y, w, b, title, x_label, y_label):
    plt.scatter(x, y, marker='o', c='b')
    plt.plot(x, w * x + b, '-r', label=f'y = {w:0.2f}x + {b:0.2f}')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.legend(loc='upper left')
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


def gradient_descent(x, y, w, b, alpha, iterations):
    cost_history = []
    for i in range(iterations):
        dj_db, dj_dw = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            cost_history.append(compute_cost_function(x, y, w, b))
        if i % math.ceil(iterations / 10) == 0:
            print(f'Iteration {i}: Cost {cost_history[-1]:8.2f}')
    return w, b, cost_history


# initialize w, b, alpha (learning rate), and iterations
# Note: change these parameters to alter model output
initial_w = 0
initial_b = 0
initial_alpha = 0.00005
initial_iterations = 10000

# plot initial data
# plot(size, price, 'Initial Data: Coffee Prices', 'size (oz)', 'price ($)')

# compute initial cost using initial parameters
initial_cost = compute_cost_function(size, price, initial_w, initial_b)
print(f'Initial Cost Function Value: {initial_cost:0.2f}')

# compute gradient
w_gradient, b_gradient = compute_gradient(size, price, initial_w, initial_b)
print(f'dj_dw: {w_gradient:0.2f}\ndj_db: {b_gradient:0.2f}')

# compute w and b using gradient descent
w_final, b_final, cost_history_final = gradient_descent(size, price, initial_w, initial_b, initial_alpha,
                                                        initial_iterations)
print(f'w_final: {w_final:0.2f}\nb_final: {b_final:0.2f}')

final_cost = compute_cost_function(size, price, w_final, b_final)
print(f'Final Cost: {final_cost:0.2f}')

# predict price for 10 oz coffee using w_final and b_final
size_10 = w_final * 10 + b_final
print(f'Price prediction for 10 oz: ${size_10:0.2f}')

if 1.50 < size_10 < 1.75:
    print('SUCCESS')
else:
    print('\nFAIL')

# plot initial data and line
plot(size, price, w_final, b_final, 'Linear Regression: Coffee Prices', 'size (oz)', 'price ($)')
