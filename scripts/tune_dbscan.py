import os
import numpy as np
from src.datasets import get_blobs, get_moons, get_circles, get_iris
from src.dbscan_tuner import DBSCANDRLTuner
from src.metrics import get_internal_metrics, print_metrics


def run_tuning_on_dataset(
    X, name, n_episodes=10, max_steps=15, verbose=True, save_dir="results/logs"
):
    print(f"\n=== [DRL-DBSCAN Tuning: {name}] ===")
    tuner = DBSCANDRLTuner(
        X, n_episodes=n_episodes, max_steps=max_steps, verbose=verbose
    )
    best_params, best_metrics = tuner.tune()
    print(f"Best params for {name}: {best_params}")
    print("Best internal metrics:")
    print_metrics(best_metrics)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}_drl_tuning.txt")
    with open(save_path, "w") as f:
        f.write(f"Best params: {best_params}\n")
        for k, v in best_metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved results to {save_path}")


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
        run_tuning_on_dataset(X, name, n_episodes=10, max_steps=15, verbose=True)
