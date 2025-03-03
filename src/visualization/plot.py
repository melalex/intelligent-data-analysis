from matplotlib import pyplot as plt


def plot_feature_count_to_f1(x, y, size=(12, 6)):
    plt.figure(figsize=size)
    plt.plot(x, y, label="F1-score")
    plt.title("Features count to F1-score")
    plt.xlabel("Features count")
    plt.ylabel("F-score")
    plt.legend()
    plt.show()
