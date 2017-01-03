# Prediction Module

from pollution_hour import PollutionHour
import numpy as np
from feed_forward_nn import process_data_set
from nn_globals import *
import scipy.stats

def predict_nn(model, input_vector, activation):
	W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
	z1 = W1.dot(input_vector) + b1
	z2 = ACTIVATION_FUNCTIONS[activation](z1) 
	return W2.dot(z2) + b2

def predict_with_feedback(model, input_vector, activation):
	W1, b1, W2, b2, U, h = model['W1'], model['b1'], model['W2'], model['b2'], model['U'], model['h']
	z1 = (W1.dot(input_vector) + b1) + U.dot(h)
	z2 = ACTIVATION_FUNCTIONS[activation](z1) 
	model['h'] = z2
	return W2.dot(z2) + b2

def predict_next_points(next_prediction, num_hours_used, input_vectors, output_vectors, future_scope):
	curr_input = input_vectors[0]
	predictions = []
	for i in range(future_scope):
		predictions.append(next_prediction(curr_input))
		if i < len(input_vectors) - 1:
			curr_input = input_vectors[i + 1]
			for k in range(i + 1):
				if k < num_hours_used:
					# depends on the fact that each input vector is [<pollutant data> + <weather data>]
					for j in range(OUTPUT_DIM):
						curr_input[-NUM_VARS * (k + 1) + j] = predictions[-k - 1][j]
	return predictions

def predict_next_nn_points(model, pollution_data, pollutant, num_hours_used, future_scope, activation, feedback = False):
	(input_vectors, output_vectors) = process_data_set([pollution_data], num_hours_used)
	sorted_pollutants = sorted(pollution_data[0].pollutants)
	if pollutant != None:
		pollutant_idx = sorted_pollutants.index(pollutant)
	if feedback:
		model['h'] = np.zeros((model['U'].shape[0], 1))
	def next_nn_prediction(curr_input):
		if not feedback:
			return predict_nn(model, curr_input, activation)
		else:
			return predict_with_feedback(model, curr_input, activation)
	predictions = predict_next_points(next_nn_prediction, num_hours_used, input_vectors, output_vectors, future_scope)
	if pollutant != None:
		return [output[pollutant_idx] for output in predictions]
	else:
		return [output[: NUM_POLLUTANTS] for output in predictions]

def predict_next_linear_points(pollution_data, pollutant, num_hours_used, future_scope):
	(input_vectors, output_vectors) = process_data_set([pollution_data], num_hours_used)
	sorted_pollutants = sorted(pollution_data[0].pollutants)
	if pollutant != None:
		pollutant_idx = sorted_pollutants.index(pollutant)
	def next_linear_prediction(curr_input):
		num_single_features = len(curr_input) / num_hours_used
		series = [[curr_input[j][0] for j in range(i, len(curr_input), num_single_features)] for i in range(num_single_features)]
		next_point = np.zeros((num_single_features, 1))
		for i in range(num_single_features):
			regression = scipy.stats.linregress(range(len(series[i])), series[i])
			slope = regression[0]
			intercept = regression[1]
			next_point[i][0] = slope * len(series[i]) + intercept
		return next_point
	predictions = predict_next_points(next_linear_prediction, num_hours_used, input_vectors, output_vectors, future_scope)
	if pollutant != None:
		return [output[pollutant_idx] for output in predictions]
	else:
		return [output[: NUM_POLLUTANTS] for output in predictions]

def predict_middle_points(pollution_data, pollutant, num_hours_used, future_scope):
	(input_vectors, output_vectors) = process_data_set([pollution_data], 2 * num_hours_used + 1)
	sorted_pollutants = sorted(pollution_data[0].pollutants)
	if pollutant != None:
		pollutant_idx = sorted_pollutants.index(pollutant)
	predictions = []
	for i in range(future_scope):
		num_single_features = input_vectors[i].shape[0] / (2 * num_hours_used + 1)
		both_series = [[input_vectors[i][k] for k \
			in range(j, len(input_vectors[i]), num_single_features)] for j \
			in range(num_single_features)]
		left_series = [series[: num_hours_used] for series in both_series]
		right_series = [series[num_hours_used + 1: ] for series in both_series]
		x_vals = range(num_hours_used) + range(num_hours_used + 1, 2 * num_hours_used + 1)
		predictions.append(np.zeros((num_single_features, 1)))
		for j in range(num_single_features):
			y_vals = left_series[j] + right_series[j]
			quad_model = np.polyfit(np.array(x_vals), np.array(y_vals), 2)
			predictions[i][j] = sum([quad_model[2 - k] * (num_hours_used ** k) for k in range(3)])
	if pollutant != None:
		return [output[pollutant_idx] for output in predictions]
	else:
		return [output[: NUM_POLLUTANTS] for output in predictions]

def isolate_pollutant_series(pollution_data, pollutant, num_hours_used, future_scope):
	sorted_pollutants = sorted(pollution_data[0].pollutants)
	if pollutant != None:
		pollutant_idx = sorted_pollutants.index(pollutant)
	(input_vectors, output_vectors) = process_data_set([pollution_data], num_hours_used)
	if pollutant != None:
		return [output[pollutant_idx] for output in output_vectors[: future_scope]]
	else:
		return output_vectors[: future_scope]
	
if __name__ == '__main__':
	main()





