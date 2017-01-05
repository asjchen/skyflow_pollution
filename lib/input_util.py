"""
Functions for parsing argparse input
"""

import argparse
import net_util
import baseline
import oracle
from nn_globals import NORM_FUNCTIONS, ACTIVATION_FUNCTIONS

default_past_scope = 12 	# number of past data points to use for prediction
default_future_scope = 12 	# number of hours in the future to predict
default_hidden_dim = 100 	# dimension of hidden layer
default_activation = 'tanh' # neural net activation function (hyperbolic tangent)
default_num_iterations = 20 # number of SGD iterations
default_norm = 'L1' 		# default norm (weighted absolute value)
default_step_scale = 0.05 	# step size factor (step_size = step_scale/sqrt(num_updates))

####################################################################
####################  ArgParse Functionality  ######################
####################################################################

def make_base_parser(parser):
	# adds arguments for all parsers
	parser.add_argument('-f', '--future_scope', type=int, \
		default=default_future_scope, \
		help='Scope of future -- number of points to predict')
	parser.add_argument('-c', '--chemical', default=None, \
		help='Pollutant to graph/output time series for')
	parser.add_argument('-n', '--norm', default=default_norm, \
		choices=NORM_FUNCTIONS.keys(), \
		help='Loss function, choose from ' + str(NORM_FUNCTIONS.keys()))

def make_baseline_parser(parser):
	# creates the baseline parser
	make_base_parser(parser)
	parser.add_argument('pollution_dir_test', type=str, \
		help='Directory with the test pollution data')
	parser.add_argument('-p', '--past_scope', type=int, \
		default=default_past_scope, \
		help='Scope of past -- number of points to interpolate from')
	parser.set_defaults(func=baseline.parse_baseline_input)

def make_oracle_parser(parser):
	# creates the oracle parser
	make_base_parser(parser)
	parser.add_argument('pollution_dir_test', type=str, \
		help='Directory with the test pollution data')
	parser.add_argument('-r', '--radius', type=int, \
		default=default_past_scope, \
		help='Radius around the target data point to interpolate from')
	parser.set_defaults(func=oracle.parse_oracle_input)

def make_nn_parser(parser):
	# creates the neural net parser
	make_base_parser(parser)
	parser.add_argument('pollution_dir_train', type=str, \
		help='Directory with the train pollution data')
	parser.add_argument('pollution_dir_test', type=str, \
		help='Directory with the test pollution data')
	parser.add_argument('-d', '--hidden_dim', type=int, \
		default=default_hidden_dim, \
		help='Dimension of the hidden layer in a 3-layer NN')
	activ_help = 'Activation function for the middle layer, choose from '
	parser.add_argument('-a', '--activation', type=str, \
		default=default_activation, \
		choices = ACTIVATION_FUNCTIONS.keys(), \
		help=activ_help + str(ACTIVATION_FUNCTIONS.keys()))
	parser.add_argument('-p', '--past_scope', type=int, \
		default=default_past_scope, \
		help='Number of hours in for each iteration of the sliding window')
	parser.add_argument('--reg_w1', type=float, default=0.0, 
		help='Regularization parameter for W1')
	parser.add_argument('--reg_b1', type=float, default=0.0, 
		help='Regularization parameter for b1')
	parser.add_argument('--reg_w2', type=float, default=0.0, 
		help='Regularization parameter for W2')
	parser.add_argument('--reg_b2', type=float, default=0.0, 
		help='Regularization parameter for b2')
	parser.add_argument('--reg_u', type=float, default=0.0, 
		help='Regularization parameter for U')
	parser.add_argument('-i', '--num_iterations', type=int, \
		default=default_num_iterations, \
		help='Number of iterations/passes over the training data')
	parser.add_argument('-s', '--step_scale', type=float, \
		default=default_step_scale, \
		help='The value with which to scale the step size')
	parser.set_defaults(func=net_util.parse_nn_input)

def make_top_parser():
	# creates all parsers and returns the top parser
	top_descr = 'Toolbox of algorithms for predicting pollution levels'
	top_parser = argparse.ArgumentParser(description=top_descr)
	subparsers = top_parser.add_subparsers(title='subcommands', \
		description='valid subcommands', \
		dest='algo', help='Algorithm types')
	baseline_descr = ('Baseline algorithm -- batch linear regression'
		' on pollution levels')
	baseline_parser = subparsers.add_parser('baseline', \
		help=baseline_descr)
	make_baseline_parser(baseline_parser)
	oracle_descr = ('Oracle algorithm -- localized quadratic regression at '
		'one point at a time, with data from both before and after '
		'each target point')
	oracle_parser = subparsers.add_parser('oracle', \
		help=oracle_descr)
	make_oracle_parser(oracle_parser)
	forward_parser = subparsers.add_parser('feed-forward', \
		help='Feed-Forward NN on sliding windows of input data')
	make_nn_parser(forward_parser)
	elman_parser = subparsers.add_parser('elman', \
		help='Elman RNN on sliding windows of input data')
	make_nn_parser(elman_parser)
	return top_parser

