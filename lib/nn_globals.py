"""
Neural Net Global Variables
"""

import numpy as np

# Global-Variables for N. Taiwan Dataset
NUM_POLLUTANTS = 8
NUM_NONPOLLUTANTS = 7

# Globals calculated from above Variables
NUM_VARS = NUM_NONPOLLUTANTS + NUM_POLLUTANTS
OUTPUT_DIM = NUM_POLLUTANTS

class NetHyperparams:
	""" This class defines all necessary hyperparameters for one run of the neural net"""

	def __init__(self, hidden_dim, activation, past_scope, reg_params, \
		num_iterations, future_scope, norm, step_scale):
		self.hidden_dim = hidden_dim			# number of hidden neurons
		self.activation = activation 			# activation function
		self.past_scope = past_scope 			# number of past data points to use for prediction
		self.reg_params = reg_params 			# regularization constants
		self.num_iterations = num_iterations  	# number of SGD iterations
		self.future_scope = future_scope  		# number of hours in the future to predict
		self.norm = norm 						# name of norm function to use
		self.step_scale = step_scale  			# in SGD: step_size = step_scale/sqrt(num_updates)

def predict_nn(model, input_vector, activation):
	# given a trained feed-forward nn model and an input, 
	# predicts the next hour output
	
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	z1 = W1.dot(input_vector) + b1
	z2 = ACTIVATION_FUNCTIONS[activation](z1) 
	return W2.dot(z2) + b2

def predict_nn_with_feedback(model, input_vector, activation):
	# given a trained recurrent nn model and an input, 
	# predicts the next hour output

	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	U, h = model['U'], model['h']
	z1 = (W1.dot(input_vector) + b1) + U.dot(h)
	z2 = ACTIVATION_FUNCTIONS[activation](z1) 
	model['h'] = z2
	return W2.dot(z2) + b2

####################################################
######## Activation Functions and Gradients ########
####################################################

def tanh(X):
	return np.tanh(X)

def tanh_grad(X):
	tanh_x = np.tanh(X)
	return 1 - np.power(tanh_x, 2)

def arctan(X):
	return np.arctan(X) * (2 / np.pi)

def arctan_grad(X):
	denom = 1 + np.power(X, 2)
	return (2 / np.pi) / denom

def relu(X):
	return np.maximum(X, 0)

def relu_grad(X):
	return np.maximum(np.sign(X), 0)

def identity(X):
	return X

def identity_grad(X):
	return np.ones(X.shape)

def softplus(X):
	return np.log(1 + np.exp(X))

def softplus_grad(X):
	return 1.0 / (1 + np.exp(-1 * X))

ACTIVATION_FUNCTIONS = { \
	'tanh': tanh, \
	'arctan': arctan, \
	'relu': relu, \
	'identity': identity, \
	'softplus': softplus, \
}
ACTIVATION_GRADIENTS = { \
	'tanh': tanh_grad, \
	'arctan': arctan_grad, \
	'relu': relu_grad, \
	'identity': identity_grad, \
	'softplus': softplus_grad, \
}

####################################################
########### Norm Functions and Gradients ###########
####################################################

def L1(predicted_levels, actual_levels, avg_levels):
	abs_diff = np.absolute(predicted_levels - actual_levels)
	return abs_diff / avg_levels

def L1_grad(predicted_levels, actual_levels, avg_levels):
	signDiff = np.sign(predicted_levels - actual_levels)
	return signDiff / avg_levels

def L2(predicted_levels, actual_levels, avg_levels):
	abs_diff = np.power(predicted_levels - actual_levels, 2)
	return (0.5 * abs_diff) / avg_levels

def L2_grad(predicted_levels, actual_levels, avg_levels):
	return (predicted_levels - actual_levels) / avg_levels

NORM_FUNCTIONS = { 'L1': L1, 'L2': L2 }
NORM_GRADIENTS = { 'L1': L1_grad, 'L2': L2_grad }





