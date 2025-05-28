import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from src.datasets import get_blobs, get_moons, get_circles, get_iris
from src.clustering import run_dbscan
from src.metrics import get_internal_metrics, print_metrics


def plot_k_distance_graph(X, k, save_path=None, show=False):
    """
    k-최근접 이웃 거리 그래프(ε 추정용)
    """
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    k_distances = np.sort(distances[:, k - 1])

    plt.figure(figsize=(6, 4))
    plt.plot(k_distances)
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{k}-th nearest neighbor distance")
    plt.title("k-distance Graph")
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()
    return k_distances


def estimate_epsilon_by_k_distance(X, minpts, plot_path=None):
    k_distances = plot_k_distance_graph(X, minpts, save_path=plot_path, show=False)
    # 상위 90~95% 구간 값(엘보우 근처) 반환
    elbow_idx = int(len(k_distances) * 0.92)
    epsilon = k_distances[elbow_idx]
    return epsilon


def run_baseline_on_dataset(
    X, name, save_dir="results/logs", plot_dir="results/figures"
):
    print(f"\n=== [Baseline DBSCAN: {name}] ===")
    n_features = X.shape[1]
    minpts = max(2, 2 * n_features)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    k_dist_path = os.path.join(plot_dir, f"{name}_k_distance.png")
    epsilon = estimate_epsilon_by_k_distance(X, minpts, plot_path=k_dist_path)
    print(f"Estimated epsilon (by k-distance): {epsilon:.4f}, minPts: {minpts}")

    labels = run_dbscan(X, eps=epsilon, min_samples=minpts)
    metrics = get_internal_metrics(X, labels)
    print("Internal metrics:")
    print_metrics(metrics)

    save_path = os.path.join(save_dir, f"{name}_baseline.txt")
    with open(save_path, "w") as f:
        f.write(f"Estimated epsilon: {epsilon}\n")
        f.write(f"minPts: {minpts}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved baseline results to {save_path}")
    print(f"k-distance graph saved to {k_dist_path}")


if __name__ == "__main__":
    datasets = {
        "blobs": get_blobs(n_samples=500, centers=3, cluster_std=0.7, random_state=42)[
            0
        ],
        "moons": get_moons(n_samples=500, noise=0.07, random_state=42)[0],
        "circles": get_circles(n_samples=500, noise=0.07, factor=0.5, random_state=42)[
            0
        ],
        "iris": get_iris()[0],
    }

    for name, X in datasets.items():
        run_baseline_on_dataset(X, name)
