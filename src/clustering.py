from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


def run_dbscan(
    X, eps=0.5, min_samples=5, metric="euclidean", scale=False, return_model=False
):
    if scale:
        X = StandardScaler().fit_transform(X)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = model.fit_predict(X)
    if return_model:
        return labels, model
    return labels


def plot_clusters(
    X,
    labels,
    title="Clustering Result",
    show_noise=True,
    figsize=(6, 5),
    save_path=None,
):
    plt.figure(figsize=figsize)
    unique_labels = set(labels)
    colors = matplotlib.colormaps["tab10"].resampled(len(unique_labels))

    for k in unique_labels:
        class_member_mask = labels == k
        if k == -1:
            # 노이즈는 검은색 점
            color = [0, 0, 0, 1]
            label = "Noise"
        else:
            color = colors(k)
            label = f"Cluster {k}"
        plt.scatter(
            X[class_member_mask, 0],
            X[class_member_mask, 1],
            s=30,
            c=[color],
            label=label,
            edgecolors="k",
            alpha=0.7,
        )

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def run_kmeans(X, n_clusters=3, scale=False, random_state=42, return_model=False):
    """
    KMeans 클러스터링 실행 (비교용)
    """
    from sklearn.cluster import KMeans

    if scale:
        X = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(X)
    if return_model:
        return labels, model
    return labels


if __name__ == "__main__":
    from src.datasets import get_moons

    X, y_true = get_moons(n_samples=300, noise=0.07)
    labels = run_dbscan(X, eps=0.3, min_samples=5)
    plot_clusters(X, labels, title="DBSCAN on Moons")

    labels_kmeans = run_kmeans(X, n_clusters=2)
    plot_clusters(X, labels_kmeans, title="KMeans on Moons")
