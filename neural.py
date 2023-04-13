import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def sigmoid(weights, bias, inputs):
    x = np.dot(inputs, weights) + bias
    return 1 / (1 + np.exp(-x))


def MSE(y_true, y_pred):
    return np.square(np.subtract(y_true, y_pred)).mean()


def plot_decision_boundary(X, y, weights, bias, title):
    plt.figure()
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label='versicolor')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label='virginica')

    x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_values = - (weights[0] * x_values + bias) / weights[1]

    plt.plot(x_values, y_values, label='Decision Boundary', color='red')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.show()


def gradient(weights, bias, X, y):
    y_pred = sigmoid(weights, bias, X)
    grad_w = np.dot(X.T, (y_pred - y)) / y.size
    grad_b = np.sum(y_pred - y) / y.size
    return grad_w, grad_b

def gradient_descent(X, y, initial_weights, initial_bias, learning_rate, num_iterations, plot_halfway):
    weights = initial_weights
    bias = initial_bias
    halfway_weights = None
    halfway_bias = None

    for i in range(num_iterations):
        grad_w, grad_b = gradient(weights, bias, X, y)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        if i == plot_halfway:
            halfway_weights = weights.copy()
            halfway_bias = bias

    return weights, bias, halfway_weights, halfway_bias


def main():
    df = pd.read_csv('irisdata.csv')
    df = df.loc[df["species"].isin(["versicolor", "virginica"])]
    df['species'] = df['species'].map({'versicolor': 0, 'virginica': 1})

    X = df.drop('species', axis=1).values
    y = df['species'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_reduced = X_scaled[:, 2:4]

    # Small error weights and bias
    small_error_weights = np.array([1, -1])
    small_error_bias = 0.5
    y_pred_small_error = sigmoid(small_error_weights, small_error_bias, X_reduced)
    mse_small_error = MSE(y, y_pred_small_error)
    print("Mean Squared Error (Small Error):", mse_small_error)
    plot_decision_boundary(X_reduced, y, small_error_weights, small_error_bias, "Small Error Decision Boundary")

    # Large error weights and bias
    large_error_weights = np.array([-1, 0.5])
    large_error_bias = 1
    y_pred_large_error = sigmoid(large_error_weights, large_error_bias, X_reduced)
    mse_large_error = MSE(y, y_pred_large_error)
    print("Mean Squared Error (Large Error):", mse_large_error)
    plot_decision_boundary(X_reduced, y, large_error_weights, large_error_bias, "Large Error Decision Boundary")

    # Gradient descent optimized weights and bias
    initial_weights = np.array([0.5, -0.5])
    initial_bias = 0

    learning_rate = 0.1
    num_iterations = 1000
    plot_halfway = 10
    optimized_weights, optimized_bias, halfway_weights, halfway_bias = gradient_descent(X_reduced, y, initial_weights, initial_bias, learning_rate, num_iterations, plot_halfway)

    y_pred_halfway = sigmoid(halfway_weights, halfway_bias, X_reduced)
    mse_halfway = MSE(y, y_pred_halfway)
    print("Mean Squared Error (Halfway through Gradient Descent):", mse_halfway)
    
    # Halfway point of gradient descent
    plot_decision_boundary(X_reduced, y, halfway_weights, halfway_bias, "Decision Boundary at Halfway Point of Gradient Descent")

    # Optimized using gradient descent
    y_pred_optimized = sigmoid(optimized_weights, optimized_bias, X_reduced)
    mse_optimized = MSE(y, y_pred_optimized)
    print("Mean Squared Error (Optimized using Gradient Descent):", mse_optimized)
    plot_decision_boundary(X_reduced, y, optimized_weights, optimized_bias, "Optimized Decision Boundary using Gradient Descent")


if __name__ == "__main__":
    main()