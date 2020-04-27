# --- default umap config -----
n_neighbors = 30
spread = 1.0
min_dist = 0.1
seed = 42

# --- default stochastic embedding config -----
# fraction of samples used as neighbors
neighbor_frac = 0.7
# fraction of samples used as centroids
centroid_frac = 0.7
smoothing_epochs = 0
smoothing_neighbors = None
max_iter = 2000