import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
from gensim.models import Word2Vec, Doc2Vec

def get_embedding(model, word):
    try:
        return model.wv[word]
    except:
        return None

def visualize_embeddings(ingredients, model, type):
    # Embed the ingredients
    embeddings = np.array([get_embedding(model, word) for word in ingredients if get_embedding(model, word) is not None])
    valid_ingredients = [word for word in ingredients if get_embedding(model, word) is not None]

    # Perform PCA to reduce dimensions for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)

    # Clustering using KMeans
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(pca_result)
    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Voronoi diagram using the centroids
    vor = Voronoi(centroids)
    fig, ax = plt.subplots(figsize=(14, 10))
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='green', line_width=2, line_alpha=0.6)

    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters, cmap='viridis', s=100)

    # Adding labels to the points
    for i, word in enumerate(valid_ingredients):
        ax.annotate(word, (pca_result[i, 0], pca_result[i, 1]), fontsize=12, ha='right')

    plt.title(f'Most Similar Words in the {type} Vocabulary')
    plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.01, aspect=40)
    plt.savefig(f"figures/{type}.png")