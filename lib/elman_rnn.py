"""
Sliding Window Elman RNN - a three-layer recurrent network that takes a 
feature vector, representing the weather and pollutant data from the past 
n hours, and outputs the predictions for the next hour's pollutant levels.

Brief description of model
------------------------------
Input:              a n-dimensional vector x_t
HLayer Input:       z_1 = W_1 x_t + b_1 + U h(t - 1)
HLayer Function:    z_2 = activation(z_1) = h(t)
HLayer Output:      z_3 = W_2 z_2 + b_2
Output:             z_3

where W_1, b_1, W_2, b_2, U are parameters to be tuned, and h(t) allows for 
the recurrence. Here, we saw that the "softmax" function performed best as
the activation function in the middle layer, with the appropriate step size.
"""

import numpy as np
from nn_globals import OUTPUT_DIM, NUM_VARS, NUM_POLLUTANTS
from nn_globals import NORM_FUNCTIONS, NORM_GRADIENTS
from nn_globals import ACTIVATION_FUNCTIONS, ACTIVATION_GRADIENTS
from pollution_hour import get_pollutants, get_variables

def calculate_loss(all_input_data, correct_output_data, model, avg_levels, \
	possible_update, hyper, print_loss_vector = False):
	""" Loss function for elman RNN

	all_input_data:			input data points
	correct_output_data:	correct output
	model:					model information (e.g., W1, W2, ...)
	avg_levels:				average levels of pollutants
	possible_update:		update function for recursive functionality 		
	hyper:					hyper parameter object	
	print_loss_vector:		True to print the loss vector	

	"""

	(W1, b1, W2, b2, U, h) = (model['W1'], model['b1'], model['W2'], \
		model['b2'], model['U'], model['h'])
	loss = 0.0

	if print_loss_vector: 
		loss_vector = np.zeros((NUM_POLLUTANTS, 1))
	for idx, X in enumerate(all_input_data):
		if X == None:
			h = np.zeros((hyper.hidden_dim, 1))
			continue
		z1 = (W1.dot(X) + b1) + U.dot(h)
		z2 = ACTIVATION_FUNCTIONS[hyper.activation](z1) 
		predicted_levels = W2.dot(z2) + b2
		normalized_vector = NORM_FUNCTIONS[hyper.norm](
			predicted_levels, correct_output_data[idx], avg_levels)
		if print_loss_vector: 
			loss_vector += normalized_vector
		loss += sum(normalized_vector)

		possible_update(model, X)
	
	if print_loss_vector:
		for i in xrange(NUM_POLLUTANTS):
			loss_vector[i] /= len(all_input_data)
		print "Loss Vector: \n", loss_vector

	# Regularization
	total_reg = 0.0
	for param in hyper.reg_params:
		if param in model:
			reg_const = hyper.reg_params[param]
			total_reg += reg_const * 0.5 * (np.linalg.norm(model[param]) ** 2) 

	return loss / len(all_input_data) + total_reg

####################################################
#################### Gradients #####################
####################################################

def loss_gradient_b2(input_data, correct_output_data, model, avg_levels, hyper):
	(W1, b1, W2, b2, U, h) = (model['W1'], model['b1'], model['W2'], \
		model['b2'], model['U'], model['h'][-1])
	gradient = np.zeros(b2.shape)
	z1 = (W1.dot(input_data) + b1) + U.dot(h)
	z2 = ACTIVATION_FUNCTIONS[hyper.activation](z1)
	predicted_levels = W2.dot(z2) + b2
	intermed_grad = NORM_GRADIENTS[hyper.norm](
		predicted_levels, correct_output_data, avg_levels)
	gradient += intermed_grad
	gradient += hyper.reg_params['b2'] * b2
	return gradient

def loss_gradient_W2(input_data, correct_output_data, model, avg_levels, hyper):
	(W1, b1, W2, b2, U, h) = (model['W1'], model['b1'], model['W2'], \
		model['b2'], model['U'], model['h'][-1])
	gradient = np.zeros(W2.shape)
	z1 = (W1.dot(input_data) + b1) + U.dot(h)
	z2 = ACTIVATION_FUNCTIONS[hyper.activation](z1)
	predicted_levels = W2.dot(z2) + b2
	j_col = NORM_GRADIENTS[hyper.norm](
		predicted_levels, correct_output_data, avg_levels)
	j_comp = np.repeat(j_col, W2.shape[1], axis=1)
	k_comp = np.repeat(np.transpose(z2), W2.shape[0], axis=0)
	gradient += np.multiply(j_comp, k_comp)
	gradient += hyper.reg_params['W2'] * W2
	return gradient

b1_cache = [] 

def dz2_db1(model, input_data, activation, i):
	if i < 0:
		return np.zeros(model['b1'].shape)

	global b1_cache
	if len(model['h']) == 1:
		b1_cache = []
	elif i < len(b1_cache):
		return b1_cache[i]

	(W1, b1, W2, b2, U, h) = (model['W1'], model['b1'], model['W2'], \
		model['b2'], model['U'], model['h'][i])
	z1 = (W1.dot(input_data) + b1) + U.dot(h) 
	z2 = ACTIVATION_FUNCTIONS[activation](z1)
	dz2_dz1 = ACTIVATION_GRADIENTS[activation](z1)

	# Base case: h = 0 so dz2/db1 = dz2/dz1 * dz1/db1 = dz2/dz1
	if i == 0:
		b1_cache.append(dz2_dz1)
		return dz2_dz1

	# Else: dz2/db1 = dz2/dz1 * dz1/db1 = dz2/dz1 * (1 + U * dz2/db1)
	diagonal_elems = np.zeros(b1.shape)
	for j in xrange(b1.shape[0]):
		ith_diag = U[j: (j + 1), j: (j + 1)]
		diagonal_elems[j] = ith_diag 
	db1_deriv = dz2_db1(model, input_data, activation, i-1)
	u_comp = dz2_dz1 * (1 + diagonal_elems * db1_deriv)
	b1_cache.append(u_comp)
	return u_comp
	

def loss_gradient_b1(input_data, correct_output_data, model, avg_levels, hyper):
	(W1, b1, W2, b2, U, h) = (model['W1'], model['b1'], model['W2'], \
		model['b2'], model['U'], model['h'][-1])
	gradient = np.zeros(b1.shape)
	z1 = (W1.dot(input_data) + b1) + U.dot(h) 
	z2 = ACTIVATION_FUNCTIONS[hyper.activation](z1)
	predicted_levels = W2.dot(z2) + b2
	mults = NORM_GRADIENTS[hyper.norm](
		predicted_levels, correct_output_data, avg_levels)
	for i in xrange(OUTPUT_DIM):
		mult_vector = np.repeat(mults[i].reshape((1, 1)), b1.shape[0], axis=0)
		z_comp = ACTIVATION_GRADIENTS[hyper.activation](z1)
		w_comp = np.transpose(W2[i: (i + 1), :])
		
		# compute (1 + U * dz2/db1) factor
		diagonal_elems = np.zeros(b1.shape)
		for k in xrange(b1.shape[0]): 
			ith_diag = U[k: (k + 1), k: (k + 1)]
			diagonal_elems[k] = ith_diag 
		# check the last argument to dz2_db1
		db1_deriv = dz2_db1(model, input_data, hyper.activation, \
			len(model['h']) - 2) 
		u_comp = 1 + diagonal_elems * db1_deriv

		# Update gradient
		gradient += mult_vector * (z_comp * (w_comp * u_comp))

	gradient += hyper.reg_params['b1'] * b1
	return gradient
 
W1_cache = [] 

def dz2_dW1(model, input_data, activation, i):
	if i < 0:
		return np.zeros(model['W1'].shape)

	global W1_cache
	if len(model['h']) == 1:
		W1_cache = []
	elif i < len(W1_cache):
		return W1_cache[i]

	(W1, b1, W2, b2, U, h) = (model['W1'], model['b1'], model['W2'], \
		model['b2'], model['U'], model['h'][i])
	z1 = (W1.dot(input_data) + b1) + U.dot(h) 
	z2 = ACTIVATION_FUNCTIONS[activation](z1)
	dz2_dz1 = ACTIVATION_GRADIENTS[activation](z1)
	k_comp = np.repeat(np.transpose(input_data), W1.shape[0], axis=0)
	j_comp = np.repeat(dz2_dz1, W1.shape[1], axis=1)

	# Base case: h = 0 so dz2/db1 = dz2/dz1 * dz1/db1 = dz2/dz1
	if i == 0:
		result = k_comp * j_comp
		W1_cache.append(result)
		return result

	# Else: dz2/db1 = dz2/dz1 * dz1/db1 = dz2/dz1 * (1 + U * dz2/db1)
	diagonal_elems = np.zeros(W1.shape)
	for k in xrange(U.shape[0]):
		kth_diag = U[k: (k + 1), k: (k + 1)]
		for l in xrange(len(input_data)):
			diagonal_elems[k][l] = kth_diag  
	dW1_deriv = dz2_dW1(model, input_data, activation, i-1)
	u_comp = j_comp * (k_comp + (diagonal_elems * dW1_deriv))
	W1_cache.append(u_comp)
	return u_comp

def loss_gradient_W1(input_data, correct_output_data, model, avg_levels, hyper):
	(W1, b1, W2, b2, U, h) = (model['W1'], model['b1'], model['W2'], \
		model['b2'], model['U'], model['h'][-1])
	gradient = np.zeros(W1.shape)
	z1 = (W1.dot(input_data) + b1) + U.dot(h) 
	z2 = ACTIVATION_FUNCTIONS[hyper.activation](z1)
	predicted_levels = W2.dot(z2) + b2
	mults = NORM_GRADIENTS[hyper.norm](
		predicted_levels, correct_output_data, avg_levels)
	for i in xrange(OUTPUT_DIM):
		mult_vector = np.tile(mults[i].reshape((1, 1)), W1.shape)
		
		# compute W_(2,i,j) * (1 - (z_(2,j))^2)
		w_col = np.transpose(W2[i: (i + 1), :])
		j_col = ACTIVATION_GRADIENTS[hyper.activation](z1) * w_col
		j_comp = np.repeat(j_col, W1.shape[1], axis=1)

		# compute x_(t,k) + U_(j,j) * dh_(t-1,j)/dW1_(j,k)
		k_comp = np.repeat(np.transpose(input_data), W1.shape[0], axis=0)
		diagonal_elems = np.zeros(W1.shape)
		for k in xrange(U.shape[0]):
			kth_diag = U[k: (k + 1), k: (k + 1)]
			for l in xrange(len(input_data)):
				diagonal_elems[k][l] = kth_diag 
		dW1_deriv = dz2_dW1(model, input_data, hyper.activation, \
			len(model['h']) - 2)
		u_comp = k_comp + (diagonal_elems * dW1_deriv)
		gradient += mult_vector * (j_comp * u_comp)
	gradient += hyper.reg_params['W1'] * W1 
	return gradient   

U_cache = []

def dz2_dU(model, input_data, activation, i):
	if i < 0:
		return np.zeros(model['U'].shape)

	global U_cache
	if len(model['h']) == 1:
		U_cache = []
	elif i < len(U_cache):
		return U_cache[i]

	(W1, b1, W2, b2, U, h) = (model['W1'], model['b1'], model['W2'], \
		model['b2'], model['U'], model['h'][i])
	z1 = (W1.dot(input_data) + b1) + U.dot(h) 
	z2 = ACTIVATION_FUNCTIONS[activation](z1)
	dz2_dz1 = ACTIVATION_GRADIENTS[activation](z1)
	k_comp = np.repeat(np.transpose(h), U.shape[0], axis=0)
	j_comp = np.repeat(dz2_dz1, U.shape[1], axis=1)

	# Base case: h = 0 so dz2/db1 = dz2/dz1 * dz1/db1 = dz2/dz1
	if i == 0:
		result = k_comp * j_comp
		U_cache.append(result)
		return result

	# Else: dz2/db1 = dz2/dz1 * dz1/db1 = dz2/dz1 * (1 + U * dz2/db1)
	diagonal_elems = np.zeros(U.shape)
	for k in xrange(U.shape[0]):
		kth_diag = U[k: (k + 1), k: (k + 1)]
		for l in xrange(U.shape[1]):
			diagonal_elems[k][l] = kth_diag  
	dU_deriv = dz2_dU(model, input_data, activation, i - 1)
	u_comp = j_comp * (k_comp + (diagonal_elems * dU_deriv))
	U_cache.append(u_comp)
	return u_comp

def loss_gradient_U(input_data, correct_output_data, model, avg_levels, hyper):
	(W1, b1, W2, b2, U, h) = (model['W1'], model['b1'], model['W2'], \
		model['b2'], model['U'], model['h'][-1])
	gradient = np.zeros(U.shape)
	z1 = (W1.dot(input_data) + b1) + U.dot(h) 
	z2 = ACTIVATION_FUNCTIONS[hyper.activation](z1)
	predicted_levels = W2.dot(z2) + b2
	mults = NORM_GRADIENTS[hyper.norm](
		predicted_levels, correct_output_data, avg_levels)
	for i in xrange(OUTPUT_DIM):
		mult_vector = np.tile(mults[i].reshape((1, 1)), U.shape)
		w_col = np.transpose(W2[i: (i + 1), :])
		j_col = ACTIVATION_GRADIENTS[hyper.activation](z1) * w_col
		j_comp = np.repeat(j_col, U.shape[1], axis=1)
		k_comp = np.repeat(np.transpose(h), U.shape[0], axis=0)
		
		diagonal_elems = np.zeros(U.shape)
		for k in xrange(U.shape[0]):
			kth_diag = U[k: (k + 1), k: (k + 1)]
			for l in xrange(U.shape[1]):
				diagonal_elems[k][l] = kth_diag 

		dU_deriv = dz2_dU(model, input_data, hyper.activation, \
			len(model['h']) - 2)
		u_comp = k_comp + (diagonal_elems * dU_deriv)

		gradient += mult_vector * (j_comp * u_comp)
	gradient += hyper.reg_params['U'] * U 
	return gradient 


####################################################
############### Processing Functions ###############
####################################################  

def process_data_set(pollution_data_list, num_hours_used):
	""" Processes data set given list of lists of pollutionHour objects
	and the number of hours used

	Returns a tuple of lists of lists (input-vectors, output-vectors);
			input-vectors is a list of lists where each entry is an input
			vector and similar for output-vectors
	"""
	input_vectors = [None]
	output_vectors = [None]
	for pollution_data in pollution_data_list:
		if len(pollution_data) <= num_hours_used:
			continue
		input_vec = []
		for i in xrange(len(pollution_data) - num_hours_used):
			input_poll = get_pollutants(pollution_data[i + num_hours_used])
			output_vectors.append(input_poll)
			if len(input_vec) == 0:
				for j in xrange(num_hours_used):
					input_vars = get_variables(pollution_data[i + j])
					input_vec += input_vars
			else:
				data_chunk = pollution_data[i + num_hours_used - 1]
				input_vars = get_variables(data_chunk)
				input_vec = input_vec[NUM_VARS: ] + input_vars
			input_vectors.append(input_vec)
		input_vectors.append(None) #indicator
		output_vectors.append(None)
	
	temp = []
	for v in input_vectors:
		if v != None:
			temp.append(np.array(v).reshape((NUM_VARS * num_hours_used, 1)))
		else:
			temp.append(None)
	input_vectors = temp

	temp = []
	for v in output_vectors:
		if v != None:
			temp.append(np.array(v).reshape((OUTPUT_DIM, 1)))
		else:
			temp.append(None)
	output_vectors = temp
	
	return (input_vectors, output_vectors)

def update(model, input_data):
	# update function for elman RNN
	W1, b1, U, h = model['W1'], model['b1'], model['U'], model['h']
	z1 = (W1.dot(input_data) + b1) + U.dot(h[-1])
	z2 = np.tanh(z1)
	model['h'].append(z2)

def get_loss_gradients():
	loss_gradients = {
		'W1': loss_gradient_W1, \
		'b1': loss_gradient_b1, \
		'W2': loss_gradient_W2, \
		'b2': loss_gradient_b2, \
		'U' : loss_gradient_U
	}
	return loss_gradients



