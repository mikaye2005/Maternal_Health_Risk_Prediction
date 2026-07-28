import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def clustering_analysis(frame, features, seed=42):
    scaled = StandardScaler().fit_transform(frame[features])
    rows = []
    labels_by_k = {}
    for k in range(2, 7):
        model = KMeans(n_clusters=k, n_init=20, random_state=seed)
        labels = model.fit_predict(scaled)
        rows.append({"clusters": k, "inertia": model.inertia_,
                     "silhouette": silhouette_score(scaled, labels)})
        labels_by_k[k] = labels
    best_k = max(rows, key=lambda row: row["silhouette"])["clusters"]
    pca = PCA(n_components=2, random_state=seed).fit_transform(scaled)
    hierarchical = AgglomerativeClustering(n_clusters=best_k).fit_predict(scaled)
    projection = pd.DataFrame({"PC1": pca[:, 0], "PC2": pca[:, 1],
                               "KMeansCluster": labels_by_k[best_k],
                               "HierarchicalCluster": hierarchical})
    return pd.DataFrame(rows), projection, best_k
