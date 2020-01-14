from scipy.special import softmax
import numpy as np

def gamma(x, y, t, axis_x=0, axis_y=0):
	N = len(t)
	c, h, w = x.shape[-3:]
	t_rep = np.reshape(t, [N, 1, 1, 1])
	t_rep = t_rep.repeat(c, 1).repeat(h, 2).repeat(w, 3)
	t_rep1 = np.reshape(t, [1, N, 1, 1, 1])
	t_rep1 = t_rep1.repeat(len(y), 0).repeat(c, 2).repeat(h, 3).repeat(w, 4)
	# vectors 10 x 3 x 32 x 32
	x_rep = np.repeat(x, N, axis_x)
	# vectors 5 x 10 x 3 x 32 x 32
	y_rep = np.repeat(y, N, axis_y)
	# vectors 5 x 10 x 3 x 32 x 32
	return x_rep*t_rep + (1-t_rep1)*y_rep

def p_ni_row(x, y, n, i):
	return gamma(x, y, (i/n), axis_x=0, axis_y=1)

def kl_divergence(p, q, axis):
	p = np.asarray(p, dtype=np.float)
	q = np.asarray(q, dtype=np.float)
	return np.sum(np.where(p != 0, p * np.log(p / q), 0), axis=axis)

def d_js(p, q, axis=1):
	p = softmax(p, axis=-1)
	q = softmax(q, axis=-1)
	m = (p + q)/2.   
	kl1 = kl_divergence(p, m, axis=axis)
	kl2 = kl_divergence(q, m, axis=axis)
	return 0.5 * (kl1 + kl2)

def d_s(x, y, axis=(1, 2, 3)):
	return np.sqrt(np.sum((x - y)**2, axis=axis))

def predict_many(model, x, n_classes, batch_size):
	# x -> (8, 5, 10, 3, 32, 32)
	orig_shape = np.shape(x)
	# x -> (40, 10, 3, 32, 32)
	x_reshape = x.reshape([-1, *orig_shape[-3:]])
	n_inputs = len(x_reshape)
	# p -> (40, 10, 10)
	preds = np.zeros([len(x_reshape), n_classes])
	
	n_batches = max(round(n_inputs//batch_size), 1)

	for b in range(n_batches):
		r1, r2 = b*batch_size, (b+1)*batch_size
		inputs = x_reshape[r1:r2]
		pred = model(inputs)
		preds[r1:r2] = pred
	
	np_preds = preds.reshape([*orig_shape[:-3], n_classes])
	return np_preds

def distance_row(model, x, y, n, batch_size, n_classes):
	y = y[:,np.newaxis]
	
	steps = np.arange(n)
	sprev = np.where(steps-1 < 0, 0, steps-1)
	
	p_prev = p_ni_row(x, y, n, sprev)
	p_i = p_ni_row(x, y, n, steps)
	
	djs = d_js(predict_many(model, p_prev, n_classes, batch_size),
			   predict_many(model, p_i, n_classes, batch_size), axis=2)
	
	# distance measure based on classification
	discriminative = np.sqrt(np.abs(djs))
	# euclidian distance measure based on structural differences
	euclidian = d_s(p_prev, p_i, axis=(2, 3, 4))
	#dist = d_p + lam * dis
	
	if (djs < 0).any():
		print('[WARNING]: Potential NaN detected.')
	
	return discriminative.sum(axis=1), euclidian.sum(axis=1)

def calculate_fisher(model, from_samples, to_samples, n, batch_size, n_classes):

	n_xs = len(from_samples)
	n_ys = len(to_samples)

	# arrays to store distance
	#  1. discriminative distance of classification
	#  2. euclidian (structural) distance in data
	discr_distances = np.zeros([n_xs, n_ys])
	eucl_distances = np.zeros([n_xs, n_ys])

	for i in range(n_xs):

		x = from_samples[i]
		x = x[np.newaxis]
		ys = to_samples[i+1:]
	
		disc_row = np.zeros(n_ys)
		eucl_row = np.zeros(n_ys)
		
		if len(ys) != 0:
			discr, euclidian = distance_row(model, x, ys, n, batch_size, n_classes)
			disc_row[i+1:] = discr
			eucl_row[i+1:] = euclidian

		discr_distances[i] = disc_row
		eucl_distances[i] = euclidian

		if (i+1) % (n_xs//5) == 0:
			print('Distance calculation %.2f %%' % (((i+1)/n_xs)*100))

	return discr_distances, eucl_distances

"""def calculate_fisher(model, xs, ys, N, lam):

	n_samples = len(samples)
	distances = np.zeros([n_samples, n_samples])

	for i in range(n_samples):

		x = samples[i]
		x = x[np.newaxis] #torch.unsqueeze(x, 0).to(device)
		ys = samples[i+1:]
	
		row_distances = np.zeros(n_samples)
		
		if len(ys) != 0:
			row_distances[i+1:] = distance_row(model, x, ys, N, lam)

		distances[i] = row_distances

		if (i+1) % (len(samples)//5) == 0:
			print('Distance calculation %.2f %%' % (((i+1)/n_samples)*100))

	distances = distances + distances.transpose()
	return distances"""
def calculate_fisher_partial(model, xs, ys, N, lam):

	n_xs = len(xs)
	n_ys = len(ys)

	distances = np.zeros([n_xs, n_ys])

	for i in range(n_xs):

		x = xs[i]
		x = x[np.newaxis] #torch.unsqueeze(x, 0).to(device)
		ys = ys[i+1:]
	
		row_distances = np.zeros(n_samples)
		
		if len(ys) != 0:
			row_distances[i+1:] = distance_row(model, x, ys, N, lam)

		distances[i] = row_distances

		if (i+1) % (len(samples)//5) == 0:
			print('Distance calculation %.2f %%' % (((i+1)/n_samples)*100))

	distances = distances + distances.transpose()
	return distances