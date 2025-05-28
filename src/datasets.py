from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np


def get_blobs(n_samples=500, centers=3, cluster_std=1.0, random_state=42, scale=True):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )
    if scale:
        X = StandardScaler().fit_transform(X)
    return X, y


def get_moons(n_samples=500, noise=0.05, random_state=42, scale=True):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    if scale:
        X = StandardScaler().fit_transform(X)
    return X, y


def get_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42, scale=True):
    X, y = make_circles(
        n_samples=n_samples, noise=noise, factor=factor, random_state=random_state
    )
    if scale:
        X = StandardScaler().fit_transform(X)
    return X, y


def get_iris(scale=True):
    data = load_iris()
    X = data.data
    y = data.target
    if scale:
        X = StandardScaler().fit_transform(X)
    return X, y


def get_dataset(name, **kwargs):
    if name == "blobs":
        return get_blobs(**kwargs)
    elif name == "moons":
        return get_moons(**kwargs)
    elif name == "circles":
        return get_circles(**kwargs)
    elif name == "iris":
        return get_iris(**kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")


if __name__ == "__main__":
    X_blobs, y_blobs = get_blobs()
    X_moons, y_moons = get_moons()
    X_circles, y_circles = get_circles()
    X_iris, y_iris = get_iris()
    print("blobs:", X_blobs.shape)
    print("moons:", X_moons.shape)
    print("circles:", X_circles.shape)
    print("iris:", X_iris.shape)
