from deepview.embeddings import create_mappings
from deepview.fisher_metric import calculate_fisher

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# N = 10
# lam = 0.00001
# no sqrt for eucl dist

class DeepView:

	def __init__(self, pred_fn, classes, max_samples, batch_size, data_shape, 
				 n=10, lam=0.0001, resolution=100, cmap='tab10', 
				 interactive=True, title='DeepView'):
		'''
		This class can be used to embed high dimensional data in
		2D. With an inverse mapping from 2D back into the sample
		space, the 2D space is sampled on a regular grid and a
		classification outcome for each point is visualized.

		TODO: 
		  * allow all data shapes
		  * make robust to different inputs and classifiers
		
		Parameters
		----------
		pred_fn	: callable, function
			Function that takes a single argument, which is data to be classified
			and returns the prediction logits of the model. 
			For an example, see the demo jupyter notebook
			https://github.com/LucaHermes/DeepView/blob/master/DeepView%20Demo.ipynb
		classes	: list, tuple of str
			All classes that the classifier uses as a list/tuple of strings. 
		max_samples	: int
			The maximum number of data samples that this class keeps for visualization.
			If the number of data samples passed via 'add_samples' exceeds this limit, 
			the oldest samples are deleted first.
		batch_size : int
			Batch size to use when calling the classifier
		data_shape : tuple, list (int)
			Shape of the input data.
		n : int
			Number of interpolations for distance calculation of two images.
		lam : float
			Weighting factor for the euclidian component of the distance calculation.
		resolution : int
			Resolution of the visualization of the classification boundaries.
		cmap : str
			Name of the colormap to use for visualization. 
			The number of distinguishable colors should correspond to n_classes.
			See here for the names: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
		'''
		self.model = pred_fn
		self.classes = classes
		self.n_classes = len(classes)
		self.max_samples = max_samples
		self.batch_size = batch_size
		self.data_shape = data_shape
		self.n = n
		self.lam = lam
		self.resolution = resolution
		self.cmap = plt.get_cmap(cmap)
		self.discr_distances = np.array([])
		self.eucl_distances = np.array([])
		self.samples = np.empty([0, *data_shape])
		self.embedded = np.empty([0, 2])
		self.y_true = np.array([])
		self.y_pred = np.array([])
		self.classifier_view = np.array([])
		self.interactive = interactive
		self.title = title
		self._init_plots()

	@property
	def num_samples(self):
		return len(self.samples)

	@property
	def distances(self):
		eucl_scale = 1. / self.eucl_distances.max()
		fisher_scale = 1. / self.discr_distances.max()
		eucl = self.eucl_distances * eucl_scale * self.lam
		fisher = self.discr_distances * fisher_scale * (1.-self.lam)
		stacked = np.dstack((fisher, eucl))
		return stacked.sum(-1)

	def reset(self):
		self.discr_distances = np.array([])
		self.eucl_distances = np.array([])
		self.samples = np.empty([0, *self.data_shape])
		self.embedded = np.empty([0, 2])
		self.y_true = np.array([])
		self.y_pred = np.array([])
		self.classifier_view = np.array([])

	def close(self):
		plt.close()

	def set_lambda(self, lam):
		#if self.lam == lam:
		#	return
		self.lam = lam
		self.update_mappings()

	def _get_plot_measures(self):
		ebd_min = np.min(self.embedded, axis=0)
		ebd_max = np.max(self.embedded, axis=0)
		ebd_extent = ebd_max - ebd_min

		# get extent of embedding
		x_min, y_min = ebd_min - 0.1 * ebd_extent
		x_max, y_max = ebd_max + 0.1 * ebd_extent
		return x_min, y_min, x_max, y_max

	def _init_plots(self):
		if self.interactive:
			plt.ion()
		self.fig, self.ax = plt.subplots(1, 1)
		self.ax.set_title(self.title)
		self.desc = self.fig.text(0.5, 0.02, '', fontsize=8, ha='center')
		self.cls_plot = self.ax.imshow(np.zeros([5, 5, 3]), 
			interpolation='gaussian', zorder=0, vmin=0, vmax=1)

		self.sample_plots = []

		for c in range(self.n_classes):
			color = self.cmap(c/(self.n_classes-1))
			plot = self.ax.plot([], [], 'o', label=self.classes[c], 
				color=color, zorder=2, picker=mpl.rcParams['lines.markersize'])
			self.sample_plots.append(plot[0])

		for c in range(self.n_classes):
			color = self.cmap(c/(self.n_classes-1))
			plot = self.ax.plot([], [], 'o', markeredgecolor=color, 
				fillstyle='none', ms=12, mew=2.5, zorder=1)
			self.sample_plots.append(plot[0])

		self.fig.canvas.mpl_connect('pick_event', self.show_sample)
		self.ax.legend()
		plt.show(block=False)

	def update_matrix(self, old_matrix, new_values, n_new, n_keep):
		to_triu = np.triu(old_matrix, k=1)
		new_mat = np.zeros([self.num_samples, self.num_samples])
		new_mat[n_new:,n_new:] = to_triu[:n_keep,:n_keep]
		new_mat[:n_new] = new_values
		# update the old distance matrix
		return new_mat + new_mat.transpose()

	def update_mappings(self):
		print('Embedding samples ...')
		self.embedded, self.inverse = create_mappings(
			self.distances, self.samples, self.data_shape)

		self.classifier_view = self.compute_grid()

	def add_samples(self, samples, labels):
		'''
		Adds samples points to the visualization.
		'''
		# prevent new samples to be bigger then maximum
		samples = samples[:self.max_samples]
		n_new = len(samples)

		# add new samples and remove depricated samples
		# and get predictions for the new samples
		self.samples = np.concatenate((samples, self.samples))[:self.max_samples]
		Y_preds = self.model(samples).argmax(axis=1)
		self.y_pred = np.concatenate((Y_preds, self.y_pred))[:self.max_samples]
		self.y_true = np.concatenate((labels, self.y_true))[:self.max_samples]

		# calculate new distances
		xs = samples
		ys = self.samples
		new_discr, new_eucl = calculate_fisher(self.model, xs, ys, self.n, 
			self.batch_size, self.n_classes)

		# add new distances
		n_keep = self.max_samples - n_new
		self.discr_distances = self.update_matrix(self.discr_distances, new_discr, n_new, n_keep)
		self.eucl_distances = self.update_matrix(self.eucl_distances, new_eucl, n_new, n_keep)

		# update mappings
		self.update_mappings()

	def compute_grid(self):
		print('Computing decision regions ...')
		# get extent of embedding
		x_min, y_min, x_max, y_max = self._get_plot_measures()
		# create grid
		xs = np.linspace(x_min, x_max, self.resolution)
		ys = np.linspace(y_min, y_max, self.resolution)
		grid = np.array(np.meshgrid(xs, ys))
		grid = np.swapaxes(grid.reshape(grid.shape[0],-1),0,1)
		
		# map gridmpoint to images
		grid_samples = self.inverse(grid)
		n_points = self.resolution**2

		mesh_preds = np.zeros([n_points, self.n_classes])

		for i in range(0, n_points, self.batch_size):
			n_preds = min(i+self.batch_size, n_points)
			batch = grid_samples[i:n_preds]
			# add epsilon for stability
			mesh_preds[i:n_preds] = self.model(batch) + 1e-8

		mesh_classes = mesh_preds.argmax(axis=1)
		mesh_max_class = max(mesh_classes)

		# get color of gridpoints
		color = self.cmap(mesh_classes/mesh_max_class)
		# scale colors by certainty
		h = -(mesh_preds*np.log(mesh_preds)).sum(axis=1)/np.log(self.n_classes)
		h = (h/h.max()).reshape(-1, 1)
		# adjust brightness
		h = np.clip(h*1.2, 0, 1)
		color = color[:,0:3]
		color = (1-h)*(0.5*color) + h*np.ones(color.shape, dtype=np.uint8)
		decision_view = color.reshape(self.resolution, self.resolution, 3)
		return decision_view

	def show_sample(self, event):
		# don't show this when the data samples are images
		if not len(self.data_shape) == 3:
			return
		# when there is an artist attribute, a 
		# concrete sample was clicked, otherwise
		# show the according synthesised image
		if hasattr(event, 'artist'):
			artist = event.artist
			ind = event.ind
			xs, ys = artist.get_data()
			point = [xs[ind][0], ys[ind][0]]
			sample, p, t = self.get_artist_sample(point)
			title = '%s <-> %s' if p != t else '%s --- %s'
			title = title % (self.classes[p], self.classes[t])

		f, a = plt.subplots()
		a.imshow(sample)
		a.set_title(title)
			
	def get_artist_sample(self, point):
		'''Maps the location of an embedded point to it's image.'''
		sample_id = np.argmin(np.linalg.norm(self.embedded - point, axis=1))
		sample = self.samples[sample_id]
		sample = sample + np.abs(sample.min())
		sample = sample / sample.max()
		yp, yt = (int(self.y_pred[sample_id]), int(self.y_true[sample_id]))
		return sample, yp, yt

	def show(self):
		x_min, y_min, x_max, y_max = self._get_plot_measures()

		self.cls_plot.set_data(self.classifier_view)
		self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
		self.ax.set_xlim((x_min, x_max))
		self.ax.set_ylim((y_min, y_max))

		params_str = 'batch size: %d - n: %d - $\lambda$: %.2f - res: %d'
		desc = params_str % (self.batch_size, self.n, self.lam, self.resolution)
		self.desc.set_text(desc)

		for c in range(self.n_classes):
			data = self.embedded[self.y_true==c]
			self.sample_plots[c].set_data(data.transpose())
			#plot = ax.plot(*data.transpose(), 'o', c=self.cmap(c/9), 
			#	label=self.classes[c])

		for c in range(self.n_classes):
			data = self.embedded[np.logical_and(self.y_pred==c, self.y_true!=c)]
			self.sample_plots[self.n_classes+c].set_data(data.transpose())
			#plot = ax.plot(*data.transpose(), 'o', markeredgecolor=self.cmap(c/9), 
			#	fillstyle='none', ms=200, linewidth=3)

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()
		#self.fig.show()
		self.fig.canvas.manager.window.raise_()
		plt.show()

	@staticmethod
	def create_simple_wrapper(classify):
		'''
		Creates a basic wrapper function to be passed
		on DeepView initialization. Works with sklearn
		predict_proba methods.

		Arguments
		---------
		classify : function
			The function of a classifier called to
			predict class probabilities. Has to return
			a vector [batch, class probabilities]

		Returns
		-------
		wrapper : function
			Wrapper function that casts inputs to numpy
			array of dtype float32.
		'''
		def wrapper(x):
			x = np.array(x, dtype=np.float32)
			pred = classify(x)
			return pred
		return wrapper
