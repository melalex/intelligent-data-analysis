from matplotlib import pyplot as plt


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


def plot_mean_shift_cluster_score(bandwidth, metric, metric_name, size=(12, 6)):
    plt.figure(figsize=size)
    plt.plot(bandwidth, metric, label=metric_name)
    plt.title(metric_name)
    plt.xlabel("Bandwidth")
    plt.ylabel(metric_name)
    plt.legend()
    plt.show()
