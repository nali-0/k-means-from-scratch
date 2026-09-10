import os
import tempfile

import imageio
import numpy as np
from matplotlib import pyplot as plt


def plot_kmeans_step(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray, iteration: int, save_path: str) -> None:
    """

    Args:
        X: ndarray of shape (n_samples, 2)
            2D data points.
        labels: ndarray of shape (n_samples,)
            Cluster assignments.
        centroids: ndarray of shape (k, 2)
            Current centroids.
        iteration: int
            Current iteration number (0-based).
        save_path: str
            Path to save the figure.
    """
    plt.figure(figsize=(6, 6))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        cluster_points = X[labels == label]
        plt.scatter(cluster_points[:, 0],
                    cluster_points[:, 1], c=[colors[i]],
                    s=30, label=f"Cluster {label}"
                    )

    plt.scatter(centroids[:, 0], centroids[:, 1],
                c="black", marker="X", s=200, edgecolors="white", linewidth=1.5,
                label="Centroids")

    plt.title(f"Iteration {iteration+1}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()

def create_gif_animation(X: np.ndarray, labels_history: list[np.ndarray], centroids_history: list[np.ndarray], output_path: str, fps: float = 1.5) -> None:
    """
    Generate a GIF animation of K-Means convergence.

    Args:
        X: ndarray of shape (n_samples, 2)
            2D data points.
        labels_history: list of ndarray
            Cluster labels at each iteration.
        centroids_history: list of ndarray
            Centroids positions at each iteration.
        output_path: str
            Path to save the animation.
        fps: float
            Frames per second.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        frames = []
        n_iters = len(labels_history)

        for i in range(n_iters):
            frame_path = os.path.join(tmpdir, f"frame_{i:03d}.png")
            plot_kmeans_step(X, labels_history[i], centroids_history[i], i, frame_path)
            frames.append(frame_path)

        with imageio.get_writer(output_path, mode="I", fps=1.5) as writer:
            for frame in frames:
                image = imageio.imread(frame)
                writer.append_data(image)

    print(f"Animation saved to {output_path}")