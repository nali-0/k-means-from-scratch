from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import tempfile

def initialize_centroids(X: np.ndarray, k: int) -> np.ndarray:
    """
    Randomly select k data points as initial centroids.

    Args:
        X: ndarray of shape (n_samples, n_features)
        k: Number of centroids.

    Returns:
        centroids: ndarray of shape (k, n_features)
    """
    
    idx = np.random.choice(X.shape[0], k, replace=False)
    return X[idx]

def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """

    Args:
        X:  ndarray of shape (n_samples, n_features) Input data.
        centroids: ndarray of shape (k, n_features) Current cluster centroids.

    Returns:
        labels: ndarray of shape (n_samples, )
            Cluster index for each point.

    """
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """
    Recompute centroids as the mean of assigned clusters.
    Args:
        X:  ndarray of shape (n_samples, n_features)
        labels: ndarray of shape (n_samples, )
            Current cluster centroids.
        k: int
            Number of centroids.

    Returns:
        new_centroids: ndarray of shape (k, n_features)
            updated cluster centroids.
    """
    new_centroids = np.zeros((k, X.shape[1]))
    for i in range(k):
        if np.sum(labels == i) > 0:
            new_centroids[i] = X[labels == i].mean(axis=0)
        else:
            new_centroids[i] = np.random.randn(X.shape[1]) * 0.1
    return new_centroids

def kmeans_manual(X: np.ndarray, k: int, max_iters: int = 15, tol: float = 1e-5)\
        -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Run K-Means clustering.

    Args:
        X: ndarray of shape (n_samples, n_features)
            Input data.
        k: int
            Number of clusters.
        max_iters: int, default = 15
            Maximum number of iterations.
        tol: float, default = 1e-5
        Convergence tolerance for centroids shift.

    Returns:
        labels_history: list of ndarray
            Cluster assignment on each iteration.
        centroids_history: list of ndarray
            Centroids position at each iteration. (with init)
    """
    centroids = initialize_centroids(X, k)
    centroids_history = [centroids.copy()]
    labels_history = []

    for iteration in range(max_iters):
        labels = assign_clusters(X, centroids)
        labels_history.append(labels.copy())

        new_centroids = update_centroids(X, labels, k)
        centroids_history.append(new_centroids.copy())

        if np.linalg.norm(new_centroids - centroids) < tol:
            print(f"Convergence on iteration: {iteration+1}")
            break

        centroids = new_centroids

    return labels_history, centroids_history

