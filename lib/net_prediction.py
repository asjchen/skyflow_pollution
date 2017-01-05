"""
Functions to predict the next few hours' worth of data
"""

import numpy as np
from feed_forward_nn import process_data_set
from nn_globals import predict_nn, predict_nn_with_feedback
from nn_globals import OUTPUT_DIM, NUM_VARS, NUM_POLLUTANTS
import scipy.stats

def predict_next_points(next_pred, test_vectors, scopes):
	# helper for predicting points given the test vectors and 
	# the prediction fucntion (as well as scopes)
	num_hours_used, future_scope = scopes
	(input_vectors, output_vectors) = test_vectors
	curr_input = input_vectors[0]
	predictions = []
	for i in xrange(future_scope):
		predictions.append(next_pred(curr_input))
		if i < len(input_vectors) - 1:
			curr_input = input_vectors[i + 1]
			for k in xrange(i + 1):
				if k < num_hours_used:
					# depends on the fact that each input vector is 
					# [<pollutant data> + <weather data>]
					for j in xrange(OUTPUT_DIM):
						update_idx = -NUM_VARS * (k + 1) + j
						curr_input[update_idx] = predictions[-k - 1][j]
	return predictions

def predict_next_nn_points(model, pollution_data, pollutant, hyper, feedback):
	""" Predicts the next points with a neural net

	model:				neural net trained model
	pollution_data:		data on which we should base our predictions
						(format: list of list of PollutionHour objects)
	pollutant:			name of pollutant whose data we want to graph
	hyper:				hyperparameter object (define in nn_globals)
	feedback: 			True for RNN

	Returns list of lists of predictions for the pollutants
	"""

	test_vectors = process_data_set([pollution_data], hyper.past_scope)
	sorted_pollutants = sorted(pollution_data[0].pollutants)
	if pollutant != None:
		pollutant_idx = sorted_pollutants.index(pollutant)
	if feedback:
		model['h'] = np.zeros((model['U'].shape[0], 1))
	def next_nn_prediction(curr_input):
		if not feedback:
			return predict_nn(model, curr_input, hyper.activation)
		else:
			return predict_nn_with_feedback(model, curr_input, hyper.activation)
	scopes = (hyper.past_scope, hyper.future_scope)
	predictions = predict_next_points(next_nn_prediction, test_vectors, scopes)
	if pollutant != None:
		return [output[pollutant_idx] for output in predictions]
	else:
		return [output[: NUM_POLLUTANTS] for output in predictions]

def predict_next_linear_points(pollution_data, pollutant, scopes):
	""" Predicts the next points with baseline linear regression.

	pollution_data:		data on which we should base our predictions
						(format: list of list of PollutionHour objects)
	pollutant:			name of pollutant whose data we want to graph
	scopes:				tuple of past and future scopes

	Returns list of lists of predictions for the pollutants
	"""

	past_scope, future_scope = scopes
	test_vectors = process_data_set([pollution_data], past_scope)
	sorted_pollutants = sorted(pollution_data[0].pollutants)
	if pollutant != None:
		pollutant_idx = sorted_pollutants.index(pollutant)
	def next_linear_prediction(curr_input):
		num_single_features = len(curr_input) / past_scope
		series = []
		for i in xrange(num_single_features):
			series.append([curr_input[j][0] for j \
				in xrange(i, len(curr_input), num_single_features)])
		next_point = np.zeros((num_single_features, 1))
		for i in xrange(num_single_features):
			x_coords = range(len(series[i]))
			regression = scipy.stats.linregress(x_coords, series[i])
			slope = regression[0]
			intercept = regression[1]
			next_point[i][0] = slope * len(series[i]) + intercept
		return next_point
	predictions = predict_next_points(
		next_linear_prediction, test_vectors, scopes)
	if pollutant != None:
		return [output[pollutant_idx] for output in predictions]
	else:
		return [output[: NUM_POLLUTANTS] for output in predictions]

def predict_middle_points(pollution_data, pollutant, scopes):
	""" Predicts the next points with oracle linear regression.

	pollution_data:		data on which we should base our predictions
						(format: list of list of PollutionHour objects)
	pollutant:			name of pollutant whose data we want to graph
	scopes:				tuple of past and future scopes

	Returns list of lists of predictions for the pollutants
	"""
	num_hours_used, future_scope = scopes
	total_scope = 2 * num_hours_used + 1
	test_vectors = process_data_set([pollution_data], total_scope)
	input_vectors, output_vectors = test_vectors
	sorted_pollutants = sorted(pollution_data[0].pollutants)
	if pollutant != None:
		pollutant_idx = sorted_pollutants.index(pollutant)
	predictions = []
	for i in xrange(future_scope):
		num_single_features = input_vectors[i].shape[0] / total_scope
		both_series = [[input_vectors[i][k] for k \
			in xrange(j, len(input_vectors[i]), num_single_features)] for j \
			in xrange(num_single_features)]
		left_series = [series[: num_hours_used] for series in both_series]
		right_series = [series[num_hours_used + 1: ] for series in both_series]
		x_vals = range(num_hours_used) + range(num_hours_used + 1, total_scope)
		predictions.append(np.zeros((num_single_features, 1)))
		for j in xrange(num_single_features):
			y_vals = left_series[j] + right_series[j]
			quad_model = np.polyfit(np.array(x_vals), np.array(y_vals), 2)
			predictions[i][j] = 0.0
			for k in xrange(3):
				predictions[i][j] += quad_model[2 - k] * (num_hours_used ** k)
	if pollutant != None:
		return [output[pollutant_idx] for output in predictions]
	else:
		return [output[: NUM_POLLUTANTS] for output in predictions]

def isolate_pollutant_series(pollution_data, pollutant, scopes):
	# isolates pollutant series out of the overall pollution data (w/weather, etc.)
	num_hours_used, future_scope = scopes
	sorted_pollutants = sorted(pollution_data[0].pollutants)
	if pollutant != None:
		poll_idx = sorted_pollutants.index(pollutant)
	test_vectors = process_data_set([pollution_data], num_hours_used)
	(input_vectors, output_vectors) = test_vectors
	if pollutant != None:
		return [output[poll_idx] for output in output_vectors[: future_scope]]
	else:
		return output_vectors[: future_scope]
	



