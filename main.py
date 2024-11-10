import math
import argparse

import numpy as np
import matplotlib.pyplot as plt


def compute_cost_function(x: np.array, y: np.array, w: float, b: float) -> float:
    """Computes the cost function.

    Args:
        x: feature vector.
        y: label vector.
        w: slope of line, also known as the "weight."
        b: y-intercept of line, also known as the "bias."

    Returns:
        cost: the prediction error. The goal is to minimize this.
    """
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        y_hat = w * x[i] + b
        cost = cost + (y_hat - y[i]) ** 2
    cost = cost / (2 * m)
    return cost


def compute_gradient(x: np.array, y: np.array, w: float, b: float):
    """Computes the gradient.

    Args:
        x: feature vector.
        y: label vector.
        w: slope of line, also known as the "weight."
        b: y-intercept of line, also known as the "bias."

    Returns:
        dj_dw: partial derivative of the cost function with respect to w.
        dj_db: partial derivative of the cost function with respect to b.
    """
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


def gradient_descent(
    x: np.array, y: np.array, w: float, b: float, alpha: float, iterations: int
):
    """Performs stochastic gradient descent to find the w and b values
    that minimizes a cost function.

    Args:
        x: feature vector.
        y: label vector.
        w: slope of line, also known as the "weight."
        b: y-intercept of line, also known as the "bias."
        alpha: learning rate.
        iterations: number of iterations.

    Returns:
        w: slope of line, also known as the "weight."
        b: y-intercept of line, also known as the "bias."
        cost_history: list of cost values.
    """
    cost_history = []
    for i in range(iterations):
        dj_db, dj_dw = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            cost_history.append(compute_cost_function(x, y, w, b))
        if i % math.ceil(iterations / 10) == 0:
            print(f"Iteration {i}: Cost {cost_history[-1]:8.2f}")
    return w, b, cost_history


def plot(
    x: np.array, y: np.array, w: float, b: float, title: str, x_label: str, y_label: str
):
    """Plot the data.

    Args:
        x: feature vector.
        y: label vector.
        w: slope of line, also known as the "weight."
        b: y-intercept of line, also known as the "bias."
        title: title of the plot.
        x_label: x-axis label of the plot.
        y_label: y-axis label of the plot.
    """
    plt.style.use("dark_background")
    plt.scatter(x, y, marker="o", c="orangered")
    plt.plot(x, w * x + b, "skyblue", label=f"y = {w:0.2f}x + {b:0.2f}")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid()
    plt.legend(loc="upper left")
    plt.show()


def predict_coffee_price(
    predict_size: float,
    initial_w: float,
    initial_b: float,
    initial_alpha: float,
    initial_iterations: int,
):
    """Use simple linear regression (SLR) with stochastic gradient descent
    to predict the price of coffee.

    Labeled data is given. The x values represent the size of coffee,
    in ounces (oz). The y values represent the price of coffee, in dollars ($).

    Args:
        predict_size: The size of coffee, in ounces (oz), subject to price prediction.
        initial_w: slope of line, also known as the "weight."
        initial_b: y-intercept of line, also known as the "bias."
        initial_alpha: learning rate.
        initial_iterations: number of iterations.
    """
    print(
        "\nLet's use simple linear regression (SLR) with stochastic gradient "
        f"descent to predict the price of a {predict_size} oz coffee!"
    )
    # Data.
    size = np.array([8, 12, 16, 20, 8, 8, 8, 8, 8, 12, 12, 12, 16, 16])
    price = np.array(
        [
            1.50,
            1.75,
            2,
            2.50,
            1.55,
            1.60,
            1.45,
            1.60,
            1.58,
            1.80,
            1.70,
            1.75,
            1.72,
            1.78,
        ]
    )

    # Compute initial cost using initial parameters.
    print("Computing the initial cost...")
    initial_cost = compute_cost_function(size, price, initial_w, initial_b)
    print(f"{initial_cost:0.2f}")

    # Compute gradient.
    print("\nComputing the gradient...")
    w_gradient, b_gradient = compute_gradient(size, price, initial_w, initial_b)
    print(f"dj_dw: {w_gradient:0.2f}\ndj_db: {b_gradient:0.2f}")

    # Compute w, b using gradient descent.
    print("\nComputing w, b values that minimize cost using gradient descent...")
    w_final, b_final, cost_history_final = gradient_descent(
        size, price, initial_w, initial_b, initial_alpha, initial_iterations
    )
    print("\nResults from gradient descent...")
    print(f"w_final: {w_final:0.2f}\nb_final: {b_final:0.2f}")
    print(f"Best fit line: y = {w_final:0.2f}x + {b_final:0.2f}")

    # Compute final cost.
    final_cost = compute_cost_function(size, price, w_final, b_final)
    print(f"Final cost: {final_cost:0.2f}")

    # Predict the price of a 10 oz. coffee using w_final, b_final.
    prediction = w_final * predict_size + b_final
    print(f"\nResults...\n{predict_size} oz coffee costs ${prediction:0.2f}.")

    # Plot the data and best fit line.
    print("\nPlotting best fit line...")
    plot(
        size,
        price,
        w_final,
        b_final,
        "Linear Regression",
        "Coffee Size (oz)",
        "Price ($)",
    )


if __name__ == "__main__":
    # Implement argument parser to retrieve coffee size from the terminal.
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=float)
    args = parser.parse_args()

    # Run the prediction.
    predict_coffee_price(args.size, 0, 0, 0.00005, 10000)
