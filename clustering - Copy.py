import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
import matplotlib.pyplot as plt

df = pd.read_csv("movies_clean.csv")

df = df[(df["budget"] > 0) & (df["revenue"] > 0)].copy()

df["roi"] = df["revenue"] / df["budget"]

feature_cols = ["budget", "revenue", "roi", "popularity", "vote_average", "vote_count"]

df_clust = df.dropna(subset=feature_cols).copy()

X = df_clust[feature_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_clust["pca_x"] = X_pca[:, 0]
df_clust["pca_y"] = X_pca[:, 1]

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_clust["cluster_kmeans"] = kmeans.fit_predict(X_scaled)

hclust = AgglomerativeClustering(n_clusters=4)
df_clust["cluster_hierarchical"] = hclust.fit_predict(X_scaled)

dbscan = DBSCAN(eps=0.7, min_samples=10)
df_clust["cluster_dbscan"] = dbscan.fit_predict(X_scaled)

spectral = SpectralClustering(
    n_clusters=4,
    affinity="nearest_neighbors",
    n_neighbors=10,
    assign_labels="kmeans",
    random_state=42
)
df_clust["cluster_graph"] = spectral.fit_predict(X_scaled)

def custom_cluster(row):
    roi = row["roi"]
    budget = row["budget"]
    revenue = row["revenue"]

    if roi > 3 and budget < 20_000_000:
        return 3
    if budget > 150_000_000 and revenue > 300_000_000:
        return 2
    if roi < 1:
        return 0
    return 1

df_clust["cluster_custom"] = df_clust.apply(custom_cluster, axis=1)

output_csv = "movie_clusters_production_scale.csv"
df_clust.to_csv(output_csv, index=False)
print(f"Saved clustered data to: {output_csv}")

algos = [
    "cluster_kmeans",
    "cluster_hierarchical",
    "cluster_dbscan",
    "cluster_graph",
    "cluster_custom"
]

for algo in algos:
    plt.figure(figsize=(7, 6))
    labels = df_clust[algo]

    scatter = plt.scatter(
        df_clust["pca_x"],
        df_clust["pca_y"],
        c=labels,
        s=20,
        cmap="tab10"
    )

    plt.title(f"PCA Scatter Plot - {algo}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()

    filename = f"{algo}_scatter.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved plot: {filename}")

print("\nDone. CSV + PNG plots generated.")
