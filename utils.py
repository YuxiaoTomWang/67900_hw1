import numpy as np
import matplotlib.pyplot as plt

# one dimensional regression
def plot_data_and_fit(X, Y, pred, x_min = None, x_max = None):
    fig, ax = plt.subplots()
    if x_min is None: x_min = np.min(X)
    if x_max is None: x_max = np.max(X)
    fineX = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
    ax.plot(fineX, pred(fineX), label='fit', color='red')
    ax.scatter(X, Y, label='data')
    ax.legend()
    plt.show()

# univariate fourier features
def fourier_features(X, k, div):
    return np.hstack([np.cos(np.pi * i * X / div) for i in range(1, k + 1)] +
                     [np.sin(np.pi * i * X / div) for i in range(1, k + 1)]
                     )

# univariate polynomial features
def polynomial_features(X, k):
    return np.hstack([X ** i for i in range(1, k + 1)])

# Assume pred returns mean and std
def plot_data_and_fit_with_stdev(X, Y, pred, x_min = None, x_max = None):
    fig, ax = plt.subplots()
    if x_min is None: x_min = np.min(X)
    if x_max is None: x_max = np.max(X)
    fineX = np.linspace(x_min, x_max, 1000).reshape(-1, 1)
    means, stdevs = pred(fineX, return_std=True)
    ax.plot(fineX, means, label='fit', color='red')
    ax.fill_between(fineX.ravel(), means - stdevs, means + stdevs, color='red', alpha=0.3)
    ax.scatter(X, Y, label='data', color='black')
    ax.legend()
    plt.show()

