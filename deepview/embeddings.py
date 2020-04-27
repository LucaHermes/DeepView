import umap
import deepview.Stochastic_Embedding as stocemb
import deepview.config as defaults
import numpy as np
import abc

def init_umap(config):
	n_neighbors = config.get('n_neighbors', defaults.n_neighbors)
	min_dist = config.get('min_dist', defaults.min_dist)
	spread = config.get('spread', defaults.spread)
	seed = config.get('seed', defaults.seed)
	return umap.UMAP(metric='precomputed', n_neighbors=n_neighbors,
		random_state=seed, spread=spread, min_dist=min_dist)

def init_inv_umap(config):
	neighbors = config.get('neighbor_frac', defaults.neighbor_frac)
	centroids = config.get('centroid_frac', defaults.centroid_frac)
	smoothing_epochs = config.get('smoothing_epochs', 
		defaults.smoothing_epochs)
	smoothing_neighbors = config.get('smoothing_neighbors', 
		defaults.smoothing_neighbors)
	max_iter = config.get('max_iter', defaults.max_iter)
	return InvMapper(neighbors, centroids, smoothing_epochs, 
		smoothing_neighbors, max_iter)

class Mapper(abc.ABC):

	def __init__(self):
		pass

	def __call__(self, x):
		return self.transform(x)

	@abc.abstractclassmethod
	def transform(self, x):
		pass

	@abc.abstractclassmethod
	def fit(self, x, y=None):
		pass

class InvMapper(stocemb.StochasticEmbedding):

	def __init__(self, neighbors, centroids, smoothing_epochs, smoothing_neighbors, max_iter):
		self.neighbors = neighbors
		self.centroids = centroids
		self.max_iter = max_iter
		super(InvMapper, self).__init__(
			n_smoothing_epochs=smoothing_epochs,
			n_smoothing_neighbors=smoothing_neighbors)

	def fit(self, x, y):
		'''
		x : embedded,
		y : x
		'''
		self.data_shape = y.shape[1:]
		flat_dim = np.prod(y.shape[1:])
		y_flat = np.reshape(y, [-1, flat_dim])

		x_min = np.min(x, axis=0)
		x_max = np.max(x, axis=0)
		av_range = np.mean(x_max - x_min)
		self.a = 500/av_range
		self.b = 1
		self.border_min_dist = av_range*1.1

		self.n_neighbors = int(len(x) * self.neighbors)
		self.n_centroids = int(len(x) * self.centroids)

		self._fit(x, y_flat, direct_adaption=True,
			eta=0.1, max_itr=self.max_iter, F=None)

	def fit_transform(self, x, y):
		self.fit(x, y)
		return self(distances)

	def transform(self, x):
		return self(x)

	def __call__(self, x):
		if not hasattr(self, 'data_shape'):
			raise AttributeError('Before calling, the \
				embedding must be trained via the fit method.')
		shape = [-1, *self.data_shape]
		map_to_sample = self._transform(x).reshape(shape)
		return map_to_sample