from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


def _cluster_description(cluster_medians: pd.DataFrame, overall_medians: pd.Series) -> pd.Series:
    spread = cluster_medians.div(overall_medians.replace(0, np.nan)).sub(1.0)
    descriptions = {}
    friendly = {
        "Age": "age",
        "SystolicBP": "systolic blood pressure",
        "DiastolicBP": "diastolic blood pressure",
        "BS": "blood sugar",
        "BodyTemp": "body temperature",
        "HeartRate": "heart rate",
    }
    for cluster, row in spread.iterrows():
        feature = row.abs().idxmax()
        direction = "higher" if row[feature] >= 0 else "lower"
        descriptions[cluster] = f"{direction.capitalize()} {friendly[feature]} profile"
    return pd.Series(descriptions, name="ProfileDescription")


def clustering_analysis(frame, features, target: str, seed: int = 42):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(frame[features])
    rows, labels_by_k = [], {}
    for k in range(2, 7):
        model = KMeans(n_clusters=k, n_init=30, random_state=seed)
        labels = model.fit_predict(scaled)
        rows.append({
            "method": "KMeans",
            "clusters": k,
            "inertia": float(model.inertia_),
            "silhouette": float(silhouette_score(scaled, labels)),
            "davies_bouldin": float(davies_bouldin_score(scaled, labels)),
        })
        labels_by_k[k] = labels

    best_k = int(max(rows, key=lambda row: row["silhouette"])["clusters"])
    kmeans_labels = labels_by_k[best_k]
    hierarchical_labels = AgglomerativeClustering(n_clusters=best_k).fit_predict(scaled)
    rows.append({
        "method": "Hierarchical",
        "clusters": best_k,
        "inertia": np.nan,
        "silhouette": float(silhouette_score(scaled, hierarchical_labels)),
        "davies_bouldin": float(davies_bouldin_score(scaled, hierarchical_labels)),
    })
    pca = PCA(n_components=2, random_state=seed).fit_transform(scaled)

    projection = pd.DataFrame({
        "PC1": pca[:, 0],
        "PC2": pca[:, 1],
        "KMeansCluster": kmeans_labels,
        "HierarchicalCluster": hierarchical_labels,
        target: frame[target].to_numpy(),
    })

    profile = frame.copy()
    profile["KMeansCluster"] = kmeans_labels
    medians = profile.groupby("KMeansCluster")[features].median().round(2)
    means = profile.groupby("KMeansCluster")[features].mean().round(2)
    means.columns = [f"{column}_Mean" for column in means.columns]
    sizes = profile.groupby("KMeansCluster").size().rename("N")
    descriptions = _cluster_description(medians, frame[features].median())
    cluster_profiles = medians.join(means).join(sizes).join(descriptions)

    risk_distribution = pd.crosstab(
        profile["KMeansCluster"],
        profile[target],
        normalize="index",
    ).reindex(columns=["low risk", "mid risk", "high risk"], fill_value=0).round(4)

    agreement = float(adjusted_rand_score(kmeans_labels, hierarchical_labels))
    return (
        pd.DataFrame(rows),
        projection,
        cluster_profiles,
        risk_distribution,
        best_k,
        agreement,
    )
