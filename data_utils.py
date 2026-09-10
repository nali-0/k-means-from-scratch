import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_and_preprocess() -> tuple[np.ndarray, np.ndarray]:
    """
    Load Iris dataset, standardize and reduce to 2D via PCA.

    Returns:
        A tuple containing:
            - X_2d ndarray of shape (150, 2) Standardized and PCA-transformed data.
            - y_true ndarray of shape (150, ) True species labels.
    """
    X, y_true = load_iris(return_X_y=True)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    x_2d = pca.fit_transform(x_scaled)

    return x_2d, y_true

def find_optimal_k(X: np.ndarray, k_range: range = range(2,10)):
    """
    Determine optimal k using Silhouette Score.

    Args:
        X: ndarray of shape (n_samples, n_features)
        k_range: Range of k values to evaluate.

    Returns:
        Number of clusters with the highest silhouette score.
        scores : dict
        Mapping k -> silhouette_score.
    """

    scores = {}

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = score
    optimal_k = max(scores, key=scores.get)
    print(f"Silhouette scores: {scores}")
    print(f"Optimal clusters: {optimal_k} (score={scores[optimal_k]:.4f})")
    return optimal_k, scores