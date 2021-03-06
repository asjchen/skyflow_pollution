"""
Sliding Window Feed Forward NN - a three-layer network that takes a feature 
vector, representing the weather and pollutant data from the past n hours, 
and outputs the predictions for the next hour's pollutant levels.

Brief description of model
------------------------------
Input:              a n-dimensional vector x
HLayer Input:       z_1 = W_1 x + b_1
HLayer Function:    z_2 = tanh(z_1) or sigmoid(z_1)
HLayer Output:      z_3 = W_2 z_2 + b_2
Output:             z_3

where W_1, b_1, W_2, b_2 are parameters to be tuned. Here, we saw that the 
"softmax" function performed best as the activation function in the middle 
layer, with the appropriate step size.
"""

import numpy as np
from nn_globals import predict_nn, OUTPUT_DIM, NUM_VARS, NUM_POLLUTANTS
from nn_globals import NORM_FUNCTIONS, NORM_GRADIENTS
from nn_globals import ACTIVATION_FUNCTIONS, ACTIVATION_GRADIENTS
from pollution_hour import get_pollutants, get_variables

def calculate_loss(all_input_data, correct_output_data, model, avg_levels, \
	possible_update, hyper, print_loss_vector = False):
	""" Loss function for feed-forward NN

	all_input_data:			input data points 
	correct_output_data:	correct output
	model:					model information (e.g., W1, W2, ...)
	avg_levels:				average levels of pollutants
	possible_update:		update function for recursive functionality 		
	hyper:					hyper parameter object	
	print_loss_vector:		True to print the loss vector	

	"""

	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	loss = 0.0

	if print_loss_vector: 
		loss_vector = np.zeros((NUM_POLLUTANTS, 1))
	for idx, X in enumerate(all_input_data):
		predicted_levels = predict_nn(model, X, hyper.activation)
		normalized_vector = NORM_FUNCTIONS[hyper.norm](
			predicted_levels, correct_output_data[idx], avg_levels)
		if print_loss_vector: 
			loss_vector = np.add(normalized_vector, loss_vector)
		loss += sum(normalized_vector)
	
	total_reg = 0.0
	for param in hyper.reg_params:
		if param in model:
			reg_const = hyper.reg_params[param]
			total_reg += reg_const * 0.5 * (np.linalg.norm(model[param]) ** 2) 

	if print_loss_vector:
		for i in xrange(NUM_POLLUTANTS):
			loss_vector[i] = loss_vector[i] / len(all_input_data)
		print "Loss _vector: \n", loss_vector

	return (loss / len(all_input_data)) + total_reg

####################################################
#################### Gradients #####################
####################################################

def loss_gradient_b2(input_data, correct_output_data, model, avg_levels, hyper):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	gradient = np.zeros(b2.shape)
	z1 = W1.dot(input_data) + b1
	z2 = ACTIVATION_FUNCTIONS[hyper.activation](z1)
	predicted_levels = W2.dot(z2) + b2 
	gradient += NORM_GRADIENTS[hyper.norm](
		predicted_levels, correct_output_data, avg_levels)
	gradient += hyper.reg_params['b2'] * b2
	return gradient

def loss_gradient_W2(input_data, correct_output_data, model, avg_levels, hyper):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	gradient = np.zeros(W2.shape)
	z1 = W1.dot(input_data) + b1
	z2 = ACTIVATION_FUNCTIONS[hyper.activation](z1)
	predicted_levels = W2.dot(z2) + b2 
	j_col = NORM_GRADIENTS[hyper.norm](
		predicted_levels, correct_output_data, avg_levels)
	j_comp = np.repeat(j_col, W2.shape[1], axis=1)
	k_comp = np.repeat(np.transpose(z2), W2.shape[0], axis=0)
	gradient += np.multiply(j_comp, k_comp)
	gradient += hyper.reg_params['W2'] * W2
	return gradient

def loss_gradient_b1(input_data, correct_output_data, model, avg_levels, hyper):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	gradient = np.zeros(b1.shape)
	z1 = W1.dot(input_data) + b1
	z2 = ACTIVATION_FUNCTIONS[hyper.activation](z1)
	predicted_levels = W2.dot(z2) + b2
	mults = NORM_GRADIENTS[hyper.norm](
		predicted_levels, correct_output_data, avg_levels)
	for i in xrange(OUTPUT_DIM):
		mult_vector = np.repeat(mults[i].reshape((1, 1)), b1.shape[0], axis=0)
		z_comp = ACTIVATION_GRADIENTS[hyper.activation](z1)
		w_comp = np.transpose(W2[i: (i + 1), :])
		gradient += mult_vector * (z_comp * w_comp)
	gradient += hyper.reg_params['b1'] * b1
	return gradient
 
def loss_gradient_W1(input_data, correct_output_data, model, avg_levels, hyper):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	gradient = np.zeros(W1.shape)
	z1 = W1.dot(input_data) + b1
	z2 = ACTIVATION_FUNCTIONS[hyper.activation](z1)
	predicted_levels = W2.dot(z2) + b2
	mults = NORM_GRADIENTS[hyper.norm](
		predicted_levels, correct_output_data, avg_levels)
	for i in xrange(OUTPUT_DIM):
		mult_vector = np.tile(mults[i].reshape((1, 1)), W1.shape)
		w_col = np.transpose(W2[i: (i + 1), :])
		j_col = ACTIVATION_GRADIENTS[hyper.activation](z1) * w_col
		j_comp = np.repeat(j_col, W1.shape[1], axis=1)
		k_comp = np.repeat(np.transpose(input_data), W1.shape[0], axis=0)
		gradient += mult_vector * (j_comp * k_comp)
	gradient += hyper.reg_params['W1'] * W1
	return gradient   

def process_data_set(pollution_data_list, num_hours_used):
	""" Processes data set given list of lists of pollutionHour objects
	and the number of hours used

	Returns a tuple of lists of lists (input-vectors, output-vectors);
			input-vectors is a list of lists where each entry is an input
			vector and similar for output-vectors
	"""
	input_vectors = []
	output_vectors = []
	for pollution_data in pollution_data_list:
		if len(pollution_data) <= num_hours_used:
			continue
		input_vec = []
		for i in xrange(len(pollution_data) - num_hours_used):
			output_vectors.append(get_pollutants(
				pollution_data[i + num_hours_used]))
			if len(input_vec) == 0:
				for j in xrange(num_hours_used):
					input_vars = get_variables(pollution_data[i + j])
					input_vec += input_vars
			else:
				input_vec = input_vec[NUM_VARS: ] + get_variables(
					pollution_data[i + num_hours_used - 1])
			input_vectors.append(input_vec)
	input_dim = NUM_VARS * num_hours_used
	input_vectors = [np.array(v).reshape((input_dim, 1)) for v \
		in input_vectors]
	output_vectors = [np.array(v).reshape((OUTPUT_DIM, 1)) for v \
		in output_vectors]
	return (input_vectors, output_vectors)

def none_func(x, y): return # used as update function for feed-forward (since that functionality is for RNN)

def get_loss_gradients():
	loss_gradients = {
		'W1': loss_gradient_W1, \
		'b1': loss_gradient_b1, \
		'W2': loss_gradient_W2, \
		'b2': loss_gradient_b2
	}
	return loss_gradients
 
