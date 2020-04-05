import scipy.spatial.distance as distan
import umap
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def leave_one_out_knn_dist_err(dists, labs, n_neighbors=5):
    nn = KNeighborsClassifier(n_neighbors=5, metric="precomputed")
    nn.fit(dists, labs) 
    
    unique_l = np.unique(labs)
    errs = 0
    # calculate the leave one out nearest neighbour error for each point
    neighs = nn.kneighbors(return_distance=False)
    neigh_labs = labs[neighs]
    counts_cl = np.zeros([labs.shape[0], unique_l.shape[0]])
    for i in range(unique_l.shape[0]):
        counts_cl[:,i] = np.sum(neigh_labs  == unique_l[i], 1)
    
    pred_labs = unique_l[np.argmax(counts_cl, 1)]
    
    # calculate the prediction error
    return sum(pred_labs != labs)/labs.shape[0]

def evaluate(deepview, X, Y):
    if len(np.shape(X)) > 2:
        bs = len(X)
        X = X.reshape(bs, -1)

    neighbors = 30
    embedding_sup = deepview.embedded
    labs = deepview.y_true
    pred_labs = deepview.y_pred
    dists = deepview.distances

    umap_unsup = umap.UMAP(n_neighbors=neighbors, random_state=11*12*13)
    embedding_unsup= umap_unsup.fit_transform(X)

    eucl_dists = distan.pdist(embedding_sup)
    eucl_dists = distan.squareform(eucl_dists)

    # calc dists in fish umap proj
    fishUmap_dists = distan.pdist(embedding_sup)
    fishUmap_dists = distan.squareform(fishUmap_dists)

    # calc dists in euclidean umap proj
    euclUmap_dists = distan.pdist(embedding_unsup)
    euclUmap_dists = distan.squareform(euclUmap_dists)

    eucl_err   = leave_one_out_knn_dist_err(eucl_dists, Y, n_neighbors=5)
    fish_err   = leave_one_out_knn_dist_err(dists, Y, n_neighbors=5)
    fishUm_err = leave_one_out_knn_dist_err(fishUmap_dists, Y, n_neighbors=5)
    euclUm_err = leave_one_out_knn_dist_err(euclUmap_dists, Y, n_neighbors=5)

    print("orig labs, knn err: eucl / fish", eucl_err, "/", fish_err)
    #print("eucl / fish / fish umap proj knn err", eucl_err, "/", fish_err, "/", fishUm_err)
    print("orig labs, knn err in proj space: eucl / fish", euclUm_err, "/", fishUm_err)


    # comparison to classifier labels
    eucl_err   = leave_one_out_knn_dist_err(eucl_dists, pred_labs, n_neighbors=5)
    fish_err   = leave_one_out_knn_dist_err(dists, pred_labs, n_neighbors=5)
    fishUm_err = leave_one_out_knn_dist_err(fishUmap_dists, pred_labs, n_neighbors=5)
    euclUm_err = leave_one_out_knn_dist_err(euclUmap_dists, pred_labs, n_neighbors=5)

    print("classif labs, knn err: eucl / fish", eucl_err, "/", fish_err)
    print("classif labs, knn acc in proj space: eucl / fish", '%.1f'%(100 -100*euclUm_err), "/", '%.1f'%(100 -100*fishUm_err))
