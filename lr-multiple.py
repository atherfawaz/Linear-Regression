import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def fetch_dataset(file_name, delimiter=','):
    dataset = pd.read_csv(file_name, delimiter, header=None)
    Y = dataset.iloc[:, -1:]
    X = dataset.iloc[:, : -1]
    return np.array(X), np.array(Y)


def normalize_data(array):
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    return np.divide(np.subtract(array, mean), std), mean, std


def cost_function(X, Y, theta):
    samples = Y.size
    cost = 0
    h = np.dot(X, theta)
    cost = (1/(2 * samples)) * \
        np.sum(np.square(np.dot(X, theta) - Y))
    return cost


def gradient_descent(X, Y, theta, learning_rate, iterations):
    print('X: ', X.shape)
    print('Y: ', Y.shape)
    print('Theta: ', theta.shape)
    count = iterations
    samples = len(Y)
    theta = theta.copy()
    cost = []
    for i in range(2):
        theta = theta - (learning_rate / samples) * \
            np.transpose(X).dot(np.dot(X, theta) - Y)
        cost.append(cost_function(X, Y, theta))
        iterations -= 1
    while iterations and (cost[count - iterations - 2] - cost[count - iterations - 1] > 0.001):
        theta = theta - (learning_rate / samples) * \
            np.transpose(X).dot(np.dot(X, theta) - Y)
        cost.append(cost_function(X, Y, theta))
        iterations -= 1
    return theta, cost


def plot_cost(cost):
    plt.plot(list(range(len(cost))), cost, '-')
    plt.title('Cost VS Iterations')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()


def main():

    # fetch dataset
    X, Y = fetch_dataset('ex1data2.txt')
    sample_count = len(Y)
    feature_count = X.shape[1]

    # norms
    X, x_mean, x_std = normalize_data(X)
    Y, y_mean, y_std = normalize_data(Y)

    # thetas
    X = np.hstack([np.ones((sample_count, 1)), X])
    theta = np.random.rand(feature_count + 1, 1)

    # hyperparameters
    learning_rate = 0.5
    iterations = 50

    # running gradient descent
    theta, cost = gradient_descent(X, Y, theta, learning_rate, iterations)

    # printing results
    print('Thetas: ', theta)
    print('Cost: ', cost[-1])
    plot_cost(cost)


if __name__ == "__main__":
    main()
