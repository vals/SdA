#!/usr/bin/env python
""" Dimensionality reduction by Stacked denoising Autoencoders
"""
import argparse
import logging

import numpy
import theano
import theano.tensor as T
import cPickle

from sda.dA import dA
from sda.utils import load_data
from sda.SdA import SdA
from sda.utils import print_array

logging.basicConfig(level=logging.INFO)


def main(args):
	logging.info('Loading data')
	# dataset = load_data(args.input)
	# X = dataset[0][0]

	X, index = load_data(args.input)

	# Compute number of minibatches
	batch_size = 10
	n_samples, n_vars = X.get_value(borrow=True).shape
	n_train_batches = n_samples / batch_size

	numpy_rng = numpy.random.RandomState(23432)

	###############
	# BUILD MODEL #
	###############
	logging.info('Building model')
	logging.info(str(n_vars) + ' -> ' + ' -> '.join(map(str, args.l)))

	sda = SdA(numpy_rng=numpy_rng, n_ins=n_vars,
			  hidden_layers_sizes=args.l, n_outs=2)

	#####################
	# PRETRAINING MODEL #
	#####################
	logging.info('Compiling training functions')
	pretraining_fns = sda.pretraining_functions(train_set_x=X,
												batch_size=batch_size)

	logging.info('Training model')
	pretraining_epochs = 15
	pretrain_lr = 0.001
	corruption_levels = [0.1, 0.2, 0.3] + [0.4] * sda.n_layers
	for i in xrange(sda.n_layers):
		for epoch in xrange(pretraining_epochs):
			c = []
			for batch_index in xrange(n_train_batches):
				c.append(pretraining_fns[i](index=batch_index,
											corruption=corruption_levels[i],
											lr=pretrain_lr))
			logging.info('Training layer {}, epoch {}, cost {}'.format(
				i, epoch, numpy.mean(c)))

	y = sda.get_lowest_hidden_values(X)
	get_y = theano.function([], y)
	y_val = get_y()
	print_array(y_val, index=index)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('input')
	parser.add_argument('-l', nargs='+', type=int, default=[2],
						help='List of hidden layer sizes, last size'
						' will be the final dimension of output')
	args = parser.parse_args()
	
	main(args)
