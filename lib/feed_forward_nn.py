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
layer.
"""

import input_util, test_util
import numpy as np
from nn_globals import *
from net_util import stochastic_gradient_descent
import net_prediction
import random

def calculate_loss(all_input_data, correct_output_data, model, avg_levels, \
	possible_update, hyper, print_loss_vector = False):
	""" Calculates the loss on the dataset (prints the loss vector)

	@param all_input_data:        list of data points in dataset where each data
								point is a list of length NUM_HOURS_USED * NUM_VARS
	@param correct_output_data:   list of lists where the ith list corresponds to the
								ith data entry in all_input_data; the lists are of length
								NUM_POLLUTANTS
	@param model:               dictionary of parameters (contains 'W1', 'b1', 'W2', 'b2')
	@param avg_levels:           list of average levels for pollutants
	@param print_loss_vector:     boolean (default false) that prints the loss 
								vector if set to true

	Returns:
		average of [sum of (|predicted - actual|/avg_level) over all pollutants] 
		over all data points
	"""
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	loss = 0.0

	if print_loss_vector: 
		loss_vector = np.zeros((NUM_POLLUTANTS, 1))
	for idx, X in enumerate(all_input_data):
		z1 = W1.dot(X) + b1
		z2 = ACTIVATION_FUNCTIONS[hyper.activation](z1) 
		predicted_levels = W2.dot(z2) + b2
		normalized_vector = NORM_FUNCTIONS[hyper.norm](
			predicted_levels, correct_output_data[idx], avg_levels)
		if print_loss_vector: 
			loss_vector = np.add(normalized_vector, loss_vector)
		loss += sum(normalized_vector)
	
	regw1 = hyper.reg_params['W1'] * 0.5 * (np.linalg.norm(W1) ** 2) 
	regw2 = hyper.reg_params['W2'] * 0.5 * (np.linalg.norm(W2) ** 2)
	regb1 = hyper.reg_params['b1'] * 0.5 * (np.linalg.norm(b1) ** 2)
	regb2 = hyper.reg_params['b2'] * 0.5 * (np.linalg.norm(b2) ** 2)

	if print_loss_vector:
		for i in range(NUM_POLLUTANTS):
			loss_vector[i] = loss_vector[i] / len(all_input_data)
		print "Loss _vector: \n", loss_vector

	return (loss / len(all_input_data)) + regw1 + regw2 + regb1 + regb2

#################### Gradient functions ####################

""" Calculate the gradient of our feed forward network with respect
	to the variables w1, w2, b1, b2

	@param input_data:           list of length NUM_HOURS_USED * NUM_VARS
								representing 1 data point           
	@param correct_output_data:   list of length NUM_POLLUTANTS representing
								the correct result for input_data
	@param model:               model params as dictionary (contains 'W1', 'W2'
								'b1', 'b2')
	@param avg_levels:           list of length NUM_POLLUTANTS representing
								the average levels of the pollutants

	Returns:
		the corresponding gradient function
"""

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
	for i in range(OUTPUT_DIM):
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
	for i in range(OUTPUT_DIM):
		mult_vector = np.tile(mults[i].reshape((1, 1)), W1.shape)
		w_col = np.transpose(W2[i: (i + 1), :])
		j_col = ACTIVATION_GRADIENTS[hyper.activation](z1) * w_col
		j_comp = np.repeat(j_col, W1.shape[1], axis=1)
		k_comp = np.repeat(np.transpose(input_data), W1.shape[0], axis=0)
		gradient += mult_vector * (j_comp * k_comp)
	gradient += hyper.reg_params['W1'] * W1
	return gradient   

##################################################################

def get_pollutants(poll_hour):
	# accessor function returns list of pollutant values in poll_hour
	return [poll_hour.pollutants[name] for name in sorted(poll_hour.pollutants)]

def get_variables(poll_hour):
	# accessor returns list of variable values in poll_hour (pollutants first)
	weather_vars = [poll_hour.weather[name] for name in sorted(poll_hour.weather)]
	return get_pollutants(poll_hour) + weather_vars

def process_data_set(pollution_data_list, num_hours_used):
	"""Parses list of pollution data into input and output vectors

	@param pollution_data_list:   list of pollutionHour objects representing all
								data in the dataset
	@param num_hours_used:      number of hours to use when predicting the next hour
	
	"""

	input_vectors = []
	output_vectors = []
	for pollution_data in pollution_data_list:
		if len(pollution_data) <= num_hours_used:
			continue
		input_vec = []
		for i in range(len(pollution_data) - num_hours_used):
			output_vectors.append(get_pollutants(
				pollution_data[i + num_hours_used]))
			if len(input_vec) == 0:
				for j in range(num_hours_used):
					input_vec = input_vec + get_variables(pollution_data[i + j])
			else:
				input_vec = input_vec[NUM_VARS: ] + get_variables(
					pollution_data[i + num_hours_used - 1])
			input_vectors.append(input_vec)
	input_vectors = [np.array(v).reshape((NUM_VARS * num_hours_used, 1)) for v \
		in input_vectors]
	output_vectors = [np.array(v).reshape((OUTPUT_DIM, 1)) for v \
		in output_vectors]
	return (input_vectors, output_vectors)

def none_func(x, y): return

def run_neural_net(pollution_data_list, hyper, verbose, verbose_n):
	""" Runs the neural net on pollution_data
	
	@param pollution_data_list:   list of pollutionHour objects representing all
								data in the dataset
	@param num_hours_used:      number of hours to use when predicting the next hour
	@param activation:          string naming activation function for the network's middle layer
	@param hidden_dim:          number of neurons to use in the hidden layer
	@param num_iterations:      number of iterations to run SGD
	@param verbose:             0 to NOT print; 1 to print on each update; 2 to print only once per verbose_n iterations
	@param verbose_n:           used only when verbose == 2 (see description of verbose param for details)       
	"""

	(input_vectors, output_vectors) = process_data_set(
		pollution_data_list, hyper.past_scope)

	train_data = zip(input_vectors, output_vectors)
	loss_gradients = [loss_gradient_W1, loss_gradient_b1, \
		loss_gradient_W2, loss_gradient_b2]
	input_dim = NUM_VARS * hyper.past_scope

	# Initialize Model
	W1 = np.random.randn(hyper.hidden_dim, input_dim) / np.sqrt(input_dim)
	b1 = np.zeros((hyper.hidden_dim, 1))
	W2 = np.random.randn(OUTPUT_DIM, hyper.hidden_dim) / np.sqrt(hyper.hidden_dim)
	b2 = np.zeros((OUTPUT_DIM, 1))
	model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

	return stochastic_gradient_descent(calculate_loss, ['W1', 'b1', 'W2', 'b2'], \
		loss_gradients, train_data, input_dim, OUTPUT_DIM, model, none_func, \
		hyper, verbose=verbose, verbose_n=verbose_n)

def test_module(pollution_dir_train, pollution_dir_test, hyper, \
	verbose = 2, verbose_n = 4):
	# TRAINING SET
	pollution_data_list_train = input_util.data_from_directory(pollution_dir_train)

	# TEST SET
	pollution_data_list_test = input_util.data_from_directory(pollution_dir_test)
	(model, loss) = run_neural_net(pollution_data_list_train, hyper, verbose, verbose_n)

	print 'PROCESSING TEST SET'

	(test_inputs, test_outputs) = process_data_set(
		pollution_data_list_test, hyper.past_scope)
	
	# Calculate average levels
	data_len = float(len(test_inputs))
	average_levels = np.sum(test_inputs, axis=0)[: OUTPUT_DIM] / data_len

	print '######################## CALCULATING LOSS ########################'
	loss = calculate_loss(test_inputs, test_outputs, model, average_levels, \
		none_func, hyper, print_loss_vector = True)
	print 'TEST LOSS: ', loss
	return model

def main():
	input_args = input_util.parse_nn_input()
	(pollution_dir_train, pollution_dir_test, hyper, pollutant) = input_args
	test_data_set = input_util.data_from_directory(pollution_dir_test)
	print 'READING DATA COMPLETE'
	model = test_module(pollution_dir_train, pollution_dir_test, hyper, \
		verbose = 2, verbose_n = 1)
	errors = np.zeros((hyper.future_scope,))
	for test_data in test_data_set:
		actual_levels = net_prediction.isolate_pollutant_series(
			test_data, pollutant, hyper.past_scope, hyper.future_scope)
		predicted_levels = net_prediction.predict_next_nn_points(
			model, test_data, pollutant, hyper.past_scope, \
			hyper.future_scope, hyper.activation)
		err = test_util.evaluate_error(
			predicted_levels, actual_levels, hyper.norm)
		for j in range(hyper.future_scope):
			errors[j] += err[j] / float(len(test_data_set))
	print 'Running Average Error'
	for i in range(len(errors)):
		print str(i + 1) + ': ' + str(sum(errors[: i + 1]) / float(i + 1))

if __name__ == '__main__':
	main()
 
