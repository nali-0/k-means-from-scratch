from data_utils import load_and_preprocess, find_optimal_k
from kmeans import kmeans_manual
from visualize import create_gif_animation


def main():
    X_2d, y_true = load_and_preprocess()

    optimal_k, scores = find_optimal_k(X_2d)

    labels_history, centroids_history = kmeans_manual(X_2d, optimal_k, max_iters=15)

    create_gif_animation(
        X_2d,
        labels_history,
        centroids_history,
        output_path="output/kmeans_iris_animation.gif"
    )

if __name__ == '__main__':
    main()