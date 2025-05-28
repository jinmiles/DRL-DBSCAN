from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
import numpy as np


def get_internal_metrics(X, labels):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    metrics = {}

    if n_clusters <= 1:
        # 클러스터가 1개 이하인 경우, 내부 지표 계산 불가
        metrics["silhouette"] = np.nan
        metrics["davies_bouldin"] = np.nan
        metrics["calinski_harabasz"] = np.nan
    else:
        metrics["silhouette"] = silhouette_score(X, labels)
        metrics["davies_bouldin"] = davies_bouldin_score(X, labels)
        metrics["calinski_harabasz"] = calinski_harabasz_score(X, labels)
    metrics["n_clusters"] = n_clusters
    return metrics


def get_external_metrics(labels_true, labels_pred):
    metrics = {}
    metrics["ARI"] = adjusted_rand_score(labels_true, labels_pred)
    metrics["NMI"] = normalized_mutual_info_score(labels_true, labels_pred)
    return metrics


def print_metrics(metrics_dict):
    for k, v in metrics_dict.items():
        print(f"{k:20s}: {v:.4f}" if isinstance(v, float) else f"{k:20s}: {v}")


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.cluster import DBSCAN

    X, y_true = make_blobs(n_samples=300, centers=3, random_state=42)
    labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)

    print("Internal metrics:")
    internal = get_internal_metrics(X, labels)
    print_metrics(internal)

    print("\nExternal metrics:")
    external = get_external_metrics(y_true, labels)
    print_metrics(external)
