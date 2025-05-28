import os
import numpy as np
from src.datasets import get_blobs, get_moons, get_circles, get_iris
from src.dbscan_tuner import DBSCANDRLTuner
from src.clustering import run_dbscan
from src.metrics import get_internal_metrics, print_metrics
from scripts.baseline import estimate_epsilon_by_k_distance


def run_full_experiment(
    X, name, n_episodes=10, max_steps=15, result_dir="results/summary", verbose=True
):
    os.makedirs(result_dir, exist_ok=True)
    n_features = X.shape[1]
    minpts = max(2, 2 * n_features)

    print(f"\n=== [DRL-DBSCAN Tuning: {name}] ===")
    tuner = DBSCANDRLTuner(
        X, n_episodes=n_episodes, max_steps=max_steps, verbose=verbose
    )
    best_params_drl, best_metrics_drl = tuner.tune()

    print(f"\n=== [Baseline DBSCAN: {name}] ===")
    epsilon_baseline = estimate_epsilon_by_k_distance(X, minpts)
    labels_baseline = run_dbscan(X, eps=epsilon_baseline, min_samples=minpts)
    metrics_baseline = get_internal_metrics(X, labels_baseline)

    print(f"\n=== [Summary: {name}] ===")
    print(
        f"{'Method':<15} | {'Epsilon':<8} | {'minPts':<6} | {'Silhouette':<10} | {'Davies-Bouldin':<14} | {'#Clusters':<9}"
    )
    print("-" * 70)
    print(
        f"{'DRL':<15} | {best_params_drl['eps']:<8.4f} | {best_params_drl['minpts']:<6} | {best_metrics_drl.get('silhouette', np.nan):<10.4f} | {best_metrics_drl.get('davies_bouldin', np.nan):<14.4f} | {best_metrics_drl.get('n_clusters', np.nan):<9}"
    )
    print(
        f"{'Baseline':<15} | {epsilon_baseline:<8.4f} | {minpts:<6} | {metrics_baseline.get('silhouette', np.nan):<10.4f} | {metrics_baseline.get('davies_bouldin', np.nan):<14.4f} | {metrics_baseline.get('n_clusters', np.nan):<9}"
    )

    save_path = os.path.join(result_dir, f"{name}_summary.txt")
    with open(save_path, "w") as f:
        f.write(
            f"{'Method':<15} | {'Epsilon':<8} | {'minPts':<6} | {'Silhouette':<10} | {'Davies-Bouldin':<14} | {'#Clusters':<9}\n"
        )
        f.write("-" * 70 + "\n")
        f.write(
            f"{'DRL':<15} | {best_params_drl['eps']:<8.4f} | {best_params_drl['minpts']:<6} | {best_metrics_drl.get('silhouette', np.nan):<10.4f} | {best_metrics_drl.get('davies_bouldin', np.nan):<14.4f} | {best_metrics_drl.get('n_clusters', np.nan):<9}\n"
        )
        f.write(
            f"{'Baseline':<15} | {epsilon_baseline:<8.4f} | {minpts:<6} | {metrics_baseline.get('silhouette', np.nan):<10.4f} | {metrics_baseline.get('davies_bouldin', np.nan):<14.4f} | {metrics_baseline.get('n_clusters', np.nan):<9}\n"
        )
    print(f"Saved summary to {save_path}")


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
        run_full_experiment(X, name, n_episodes=10, max_steps=15, verbose=True)
