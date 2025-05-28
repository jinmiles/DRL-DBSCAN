import numpy as np
from src.datasets import get_blobs
from src.metrics import (
    get_internal_metrics,
    get_external_metrics,
)


def test_internal_metrics():
    X, y_true = get_blobs(n_samples=100, centers=3, cluster_std=0.5, random_state=0)
    labels_perfect = y_true
    metrics = get_internal_metrics(X, labels_perfect)
    assert metrics["n_clusters"] == 3
    assert 0.0 <= metrics["silhouette"] <= 1.0 or np.isnan(metrics["silhouette"])
    assert metrics["davies_bouldin"] >= 0.0 or np.isnan(metrics["davies_bouldin"])
    assert metrics["calinski_harabasz"] >= 0.0 or np.isnan(metrics["calinski_harabasz"])
    print("test_internal_metrics passed.")


def test_external_metrics():
    X, y_true = get_blobs(n_samples=100, centers=3, cluster_std=0.5, random_state=0)
    labels_bad = np.random.permutation(y_true)
    metrics_true = get_external_metrics(y_true, y_true)
    metrics_bad = get_external_metrics(y_true, labels_bad)
    assert metrics_true["ARI"] == 1.0
    assert metrics_true["NMI"] == 1.0
    assert -1.0 <= metrics_bad["ARI"] <= 1.0
    assert 0.0 <= metrics_bad["NMI"] <= 1.0
    print("test_external_metrics passed.")


def test_edge_cases():
    X, y_true = get_blobs(n_samples=10, centers=1, cluster_std=0.1, random_state=0)
    labels_noise = -1 * np.ones(10, dtype=int)
    metrics = get_internal_metrics(X, labels_noise)
    assert metrics["n_clusters"] == 0
    assert np.isnan(metrics["silhouette"])
    print("test_edge_cases (all noise) passed.")

    labels_one = np.zeros(10, dtype=int)
    metrics = get_internal_metrics(X, labels_one)
    assert metrics["n_clusters"] == 1
    assert np.isnan(metrics["silhouette"])
    print("test_edge_cases (one cluster) passed.")


if __name__ == "__main__":
    test_internal_metrics()
    test_external_metrics()
    test_edge_cases()
    print("All metric tests passed.")
