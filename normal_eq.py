import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def fetch_dataset(file_name, delimiter=','):
    dataset = pd.read_csv(file_name, delimiter, header=None)
    Y = dataset.iloc[:, -1:]
    X = dataset.iloc[:, : -1]
    return np.array(X), np.array(Y)


def normal_equation(X, Y):
    temp = np.dot(np.transpose(X), X)
    temp = np.linalg.inv(temp)
    temp2 = np.dot(np.transpose(X), Y)
    return np.dot(temp, temp2)


def predict(features, theta):
    features = np.hstack([np.ones(1), features])
    return np.dot(features, theta)


def main():
    X, Y = fetch_dataset('ex1data2.txt')
    sample_count = len(Y)
    X = np.hstack([np.ones((sample_count, 1)), X])
    feature_count = X.shape[1]
    theta = normal_equation(X, Y)
    features = np.array([1650, 3])
    print(theta.shape)
    prediction = predict(features, theta)
    print('Prediction: ', prediction)


if __name__ == "__main__":
    main()
