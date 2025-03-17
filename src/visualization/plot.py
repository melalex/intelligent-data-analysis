from matplotlib import pyplot as plt
import numpy as np
import torch

from src.util.noise import add_noise


def plot_feature_count_to_f1(x, y, size=(12, 6)):
    plt.figure(figsize=size)
    plt.plot(x, y, label="F1-score")
    plt.title("Features count to F1-score")
    plt.xlabel("Features count")
    plt.ylabel("F-score")
    plt.legend()
    plt.show()


def plot_2d_data(x, y, size=(12, 6)):
    plt.figure(figsize=size)
    scatter = plt.scatter(x[:, 0], x[:, 1], c=y, cmap="viridis", edgecolors="k")
    plt.colorbar(scatter, label="Class Label")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("2D Data Visualization with Colors")
    plt.show()


def plot_2d_data(data, size=(12, 6)):
    x, y = data[:, 0], data[:, 1]

    plt.figure(figsize=size)
    plt.scatter(x, y, c="r", marker="o", label="Data Points")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("2D Data Visualization")
    plt.show()


def plot_3d_data(data, size=(12, 6)):
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c="r", marker="o", label="Data Points")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("3D Scatter Plot of Given Data")
    ax.legend()
    plt.show()


def plot_1d_data(data):
    plt.scatter(
        np.arange(len(data)),
        data,
        color="r",
        marker="o",
        label="1D Points",
    )

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("1D Array of Points")
    plt.legend()
    plt.show()


def plot_mean_shift_cluster_score(bandwidth, metric, metric_name, size=(12, 6)):
    plt.figure(figsize=size)
    plt.plot(bandwidth, metric, label=metric_name)
    plt.title(metric_name)
    plt.xlabel("Bandwidth")
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()


def plot_images(x):
    for i in range(len(x)):
        plt.subplot(330 + 1 + i)
        plt.imshow(x[i], cmap="gray")
    plt.show()


def plot_tensors(x):
    tensors = [torch.reshape(it, (28, 28)) for it in x]
    plot_images(tensors)


def sample_random_images(x):
    return [x[np.random.randint(0, len(x))] for _ in range(9)]
