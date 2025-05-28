import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples


def load_results(result_dir, name):
    drl_path = os.path.join(result_dir, f"{name}_drl_tuning.txt")
    base_path = os.path.join(result_dir, f"{name}_baseline.txt")

    def parse_metrics(path):
        metrics = {}
        with open(path, "r") as f:
            for line in f:
                if ":" in line:
                    k, v = line.strip().split(":", 1)
                    try:
                        metrics[k.strip()] = float(v.strip())
                    except:
                        metrics[k.strip()] = v.strip()
        return metrics

    return parse_metrics(drl_path), parse_metrics(base_path)


def plot_metrics_bar_grayscale(
    drl_metrics, base_metrics, name, save_dir="results/figures"
):
    metrics = ["silhouette", "davies_bouldin", "calinski_harabasz", "n_clusters"]
    drl_values = [drl_metrics.get(m, np.nan) for m in metrics]
    base_values = [base_metrics.get(m, np.nan) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    plt.style.use("grayscale")
    plt.figure(figsize=(8, 5))
    plt.bar(
        x - width / 2,
        drl_values,
        width,
        color="dimgray",
        hatch="///",
        label="DRL-DBSCAN",
        edgecolor="black",
        linewidth=1.5,
    )
    plt.bar(
        x + width / 2,
        base_values,
        width,
        color="lightgray",
        hatch="...",
        label="Baseline",
        edgecolor="black",
        linewidth=1.5,
    )
    plt.xticks(x, metrics)
    plt.ylabel("Score")
    plt.title(f"Clustering Quality Comparison (Grayscale): {name}")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{name}_metrics_bar_grayscale.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_clusters_2d_grayscale(
    X, labels_list, method_names, name, save_dir="results/figures"
):
    markers = ["o", "s", "^", "x", "D", "*", "v", "P"]
    grays = ["black", "dimgray", "gray", "darkgray", "silver", "lightgray"]
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    plt.style.use("grayscale")
    plt.figure(figsize=(10, 4))
    for i, (labels, mname) in enumerate(zip(labels_list, method_names)):
        plt.subplot(1, len(labels_list), i + 1)
        unique_labels = sorted(set(labels))
        for idx, k in enumerate(unique_labels):
            mask = labels == k
            color = grays[idx % len(grays)]
            marker = markers[idx % len(markers)]
            label = f"Cluster {k}" if k != -1 else "Noise"
            plt.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                s=30 if k != -1 else 10,
                c=color,
                marker=marker,
                label=label,
                edgecolors="black",
                alpha=0.8 if k != -1 else 0.5,
                linewidths=0.7,
            )
        plt.title(mname)
        plt.axis("off")
    plt.suptitle(f"Clustering Result (PCA, Grayscale): {name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f"{name}_clusters_pca_grayscale.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_silhouette_grayscale(X, labels, name, method, save_dir="results/figures"):
    if len(set(labels)) <= 1:
        print(f"Silhouette plot skipped for {method} (only one cluster).")
        return
    sil_vals = silhouette_samples(X, labels)
    y_lower = 10
    plt.style.use("grayscale")
    plt.figure(figsize=(6, 4))
    for i, k in enumerate(np.unique(labels)):
        ith_cluster_silhouette_values = sil_vals[labels == k]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = str(0.2 + 0.6 * i / max(1, len(np.unique(labels))))  # 0=검정, 1=흰색
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor="black",
            alpha=0.7,
        )
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(k), fontsize=9)
        y_lower = y_upper + 10
    plt.xlabel("Silhouette coefficient values")
    plt.ylabel("Cluster label")
    plt.title(f"Silhouette plot ({method}, Grayscale): {name}")
    plt.axvline(np.mean(sil_vals), color="black", linestyle="--")
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{name}_silhouette_{method}_grayscale.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    from src.datasets import get_blobs, get_moons, get_circles, get_iris
    from src.clustering import run_dbscan

    datasets = {
        "blobs": get_blobs(n_samples=500, centers=3, cluster_std=0.7, random_state=42),
        "moons": get_moons(n_samples=500, noise=0.07, random_state=42),
        "circles": get_circles(n_samples=500, noise=0.07, factor=0.5, random_state=42),
        "iris": get_iris(),
    }
    result_dir = "results/logs"
    save_dir = "results/figures"
    os.makedirs(save_dir, exist_ok=True)

    for name, (X, y_true) in datasets.items():
        drl_metrics, base_metrics = load_results(result_dir, name)
        plot_metrics_bar_grayscale(drl_metrics, base_metrics, name, save_dir)

        from src.dbscan_tuner import DBSCANDRLTuner

        best_params_drl = {
            "eps": drl_metrics.get("eps", 0.5),
            "minpts": int(drl_metrics.get("minpts", 5)),
        }
        labels_drl = run_dbscan(
            X, eps=best_params_drl["eps"], min_samples=best_params_drl["minpts"]
        )
        epsilon_base = base_metrics.get("Estimated epsilon", 0.5)
        minpts_base = int(base_metrics.get("minPts", 5))
        labels_base = run_dbscan(X, eps=epsilon_base, min_samples=minpts_base)

        plot_clusters_2d_grayscale(
            X, [labels_drl, labels_base], ["DRL-DBSCAN", "Baseline"], name, save_dir
        )
        plot_silhouette_grayscale(X, labels_drl, name, "DRL", save_dir)
        plot_silhouette_grayscale(X, labels_base, name, "Baseline", save_dir)
