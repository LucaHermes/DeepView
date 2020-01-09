from deepview.embeddings import create_mappings
from deepview.fisher_metric import calculate_fisher

import matplotlib.pyplot as plt
from scipy.special import softmax
import numpy as np

# N = 10
# lam = 0.00001
# no sqrt for eucl dist

class DeepView:

	def __init__(self, pred_fn, classes, max_samples, batch_size, img_shape, 
				 img_channels, n=10, lam=0.0001, resolution=100, cmap='tab10'):
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
		img_shape : tuple, list (int)
			Width and height of the input samples.
		img_channels : int
			Number of image channels.
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
		self.img_shape = img_shape
		self.img_channels = img_channels
		self.n = n
		self.lam = lam
		self.resolution = resolution
		self.cmap = plt.get_cmap(cmap)
		self.distances = np.array([])
		self.samples = np.empty([0, img_channels, *img_shape])
		self.embedded = np.empty([0, 2])
		self.y_true = np.array([])
		self.y_pred = np.array([])
		self.classifier_view = np.array([])
		self._init_plots()

	@property
	def num_samples(self):
		return len(self.samples)

	def _get_plot_measures(self):
		ebd_min = np.min(self.embedded, axis=0)
		ebd_max = np.max(self.embedded, axis=0)
		ebd_extent = ebd_max - ebd_min

		# get extent of embedding
		x_min, y_min = ebd_min - 0.1 * ebd_extent
		x_max, y_max = ebd_max + 0.1 * ebd_extent
		return x_min, y_min, x_max, y_max

	def _init_plots(self):
		plt.ion()
		self.fig, self.ax = plt.subplots(1, 1)
		self.ax.set_title('DeepView')
		self.cls_plot = self.ax.imshow(np.zeros([5, 5, 3]), 
			interpolation='gaussian', zorder=0, vmin=0, vmax=1)

		self.sample_plots = []

		for c in range(self.n_classes):
			color = self.cmap(c/(self.n_classes-1))
			plot = self.ax.plot([], [], 'o', label=self.classes[c], 
				color=color, zorder=2)
			self.sample_plots.append(plot[0])

		for c in range(self.n_classes):
			color = self.cmap(c/(self.n_classes-1))
			plot = self.ax.plot([], [], 'o', markeredgecolor=color, 
				fillstyle='none', ms=12, mew=2.5, zorder=1)
			self.sample_plots.append(plot[0])

		self.ax.legend()
		plt.show(block=False)

	def to_probs(self, logits):
		return softmax(logits, axis=-1)

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
		Y_preds = self.to_probs(self.model(samples)).argmax(axis=1)
		self.y_pred = np.concatenate((Y_preds, self.y_pred))[:self.max_samples]
		self.y_true = np.concatenate((labels, self.y_true))[:self.max_samples]

		# calculate new distances
		xs = samples
		ys = self.samples
		new_distances = calculate_fisher(self.model, xs, ys, self.n, self.lam, 
			self.batch_size, self.n_classes)

		# add new distances
		old_distances = np.triu(self.distances, k=1)
		distances = np.zeros([self.num_samples, self.num_samples])
		n_old = len(self.distances)

		keep_dists = self.max_samples - n_new
		distances[n_new:,n_new:] = old_distances[:keep_dists,:keep_dists]
		distances[:n_new] = new_distances
		self.distances = distances + distances.transpose()

		# update mappings
		print('Embedding samples ...')
		self.embedded, self.inverse = create_mappings(
			self.distances, self.samples, self.img_shape, self.img_channels)

		self.classifier_view = self.compute_grid()

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
		grid_imgs = self.inverse(grid)
		n_points = self.resolution**2

		mesh_preds = np.zeros([n_points, self.n_classes])

		for i in range(0, n_points, self.batch_size):
			n_preds = min(i+self.batch_size, n_points)
			batch = grid_imgs[i:n_preds]
			mesh_preds[i:n_preds] = self.to_probs(self.model(batch))

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

	def show(self):
		x_min, y_min, x_max, y_max = self._get_plot_measures()

		#ax.imshow(decision_view, 
		#	extent=(x_min, x_max, y_max, y_min), 
		#	interpolation='gaussian')
		self.cls_plot.set_data(self.classifier_view)
		self.cls_plot.set_extent((x_min, x_max, y_max, y_min))

		for c in range(self.n_classes):
			data = self.embedded[self.y_pred==c]
			self.sample_plots[c].set_data(data.transpose())
			#plot = ax.plot(*data.transpose(), 'o', c=self.cmap(c/9), 
			#	label=self.classes[c])

		for c in range(self.n_classes):
			data = self.embedded[np.logical_and(self.y_pred!=c, self.y_true==c)]
			self.sample_plots[self.n_classes+c].set_data(data.transpose())
			#plot = ax.plot(*data.transpose(), 'o', markeredgecolor=self.cmap(c/9), 
			#	fillstyle='none', ms=200, linewidth=3)

		self.fig.canvas.draw()
		self.fig.canvas.flush_events()
		self.fig.show()
		self.fig.canvas.manager.window.raise_()
		#plt.show()