import matplotlib.pyplot as plt
import numpy as np

def plot_feature_distributions(X, feature_names=None):
    # Plotting histograms for each feature.
    for i in range(X.shape[1]):
        plt.hist(X[:, i], bins=30)
        title = feature_names[i] if feature_names else f"Feature {i}"
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()


def plot_correlation_heatmap(corr_matrix, feature_names=None):
    corr_matrix = np.asarray(corr_matrix, dtype=float)

    plt.imshow(corr_matrix, cmap="coolwarm")
    plt.colorbar()

    if feature_names:
        plt.xticks(range(len(feature_names)), feature_names, rotation=90)
        plt.yticks(range(len(feature_names)), feature_names)

    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()
