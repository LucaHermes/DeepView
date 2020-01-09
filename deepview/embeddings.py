import umap
import deepview.Stochastic_Embedding as stocemb
import numpy as np

def embed(distances, seed=42):
	N_NEIGBORS = 15

	mapper = umap.UMAP(metric="precomputed", n_neighbors=N_NEIGBORS, 
                         random_state=seed, spread=2, min_dist=1)


	mapper = mapper.fit(distances) 
	embedding = mapper.transform(distances)

	return embedding

def get_inverse_mapper(samples, embedded, image_size, channels, n_bins=100):
	SCALE = 1.1

	x_flat = np.reshape(samples, [-1, image_size**2 * channels])
	
	ebd_min = np.min(embedded, axis=0)
	ebd_max = np.max(embedded, axis=0)
	av_range = np.mean(ebd_max - ebd_min)

	embd = stocemb.StochasticEmbedding(
		n_centroids=10, n_smoothing_epochs=0, 
		n_neighbors=len(samples),
		a=10, b=1, border_min_dist=av_range*SCALE*1.05)
	
	embd.fit(embedded, x_flat, direct_adaption=True, eta=0.1, max_itr=2000, F=None)

	return embd

def create_mappings(distances, samples, image_size, channels):
	embedded = embed(distances)
	inv = get_inverse_mapper(samples, embedded, image_size, channels)

	img_shape = [-1, channels, image_size, image_size]
	map_to_img = lambda ebd: inv.transform(ebd).reshape(img_shape)

	return embedded, map_to_img