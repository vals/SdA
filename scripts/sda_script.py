""" Apply the autoencoding strategy on my own data
"""

import numpy
import theano
import theano.tensor as T
import cPickle

from sda.dA import dA
from sda.logistic_sgd import load_data
from sda.SdA import SdA
from sda.utils import print_array


def apply_dA():
	# Load data from file
	dataset = load_data('/Users/val/Documents/cell-cycle/top_expression.pkl.gz')
	X = dataset[0][0]

	# compute number of minibatches for training, validation and testing
	batch_size = 15
	n_train_batches = X.get_value(borrow=True).shape[0] / batch_size

	# allocate symbolic variables for the data
	index = T.lscalar()  # index to a [mini]batch
	x = T.matrix('x')

	###############
	# BUILD MODEL #
	###############

	rng = numpy.random.RandomState(123)
	da = dA(numpy_rng=rng, input=x, n_visible=2000, n_hidden=2)

	cost, updates = da.get_cost_updates(corruption_level=0.,
										learning_rate=0.1)

	train_da = theano.function([index], cost, updates=updates,
		givens={x: X[index * batch_size:(index + 1) * batch_size]})

	###############
	# TRAIN MODEL #
	###############

	# go through training epochs
	training_epochs = 15
	for epoch in xrange(training_epochs):
		c = []
		for batch_index in xrange(n_train_batches):
			c.append(train_da(batch_index))

		print 'Traning epoch {}, cost {}'.format(epoch, numpy.mean(c))

	y = da.get_hidden_values(X)
	get_y = theano.function([], y)
	y_val = get_y()
	print y_val.T
	print y_val.shape

	with open('out.pkl', 'wb') as pkl:
		cPickle.dump(da.W.get_value(borrow=True), pkl)

	print 'Saved output'


def apply_SdA():
	# dataset = load_data('/Users/val/Documents/cell-cycle/top_expression.pkl.gz')
	dataset = load_data('/Users/val/Documents/cell-cycle/test.pkl.gz')
	X = dataset[0][0]

	# Compute number of minibatches
	batch_size = 10
	n_train_batches, n_vars = X.get_value(borrow=True).shape
	n_train_batches /= batch_size

	numpy_rng = numpy.random.RandomState(23432)

	###############
	# BUILD MODEL #
	###############
	print('... building the model')

	sda = SdA(numpy_rng=numpy_rng, n_ins=n_vars,
			  hidden_layers_sizes=[500, 125, 2], n_outs=2)

	#####################
	# PRETRAINING MODEL #
	#####################
	print '... getting the pretraining functions'
	pretraining_fns = sda.pretraining_functions(train_set_x=X,
												batch_size=batch_size)

	print('... pre-training the model')
	pretraining_epochs = 15
	pretrain_lr=0.001
	corruption_levels = [0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4]
	for i in xrange(sda.n_layers):
		for epoch in xrange(pretraining_epochs):
			c = []
			for batch_index in xrange(n_train_batches):
				c.append(pretraining_fns[i](index=batch_index,
											corruption=corruption_levels[i],
											lr=pretrain_lr))
			print 'Pre-training layer {}, epoch {}, cost {}'.format(
				i, epoch, numpy.mean(c))

	y = sda.get_lowest_hidden_values(X)
	get_y = theano.function([], y)
	y_val = get_y()
	print_array(y_val)

	with open('out.pkl', 'wb') as pkl:
		cPickle.dump(y_val, pkl)

	print "... Results saved to out.pkl"


if __name__ == '__main__':
	apply_SdA()
