"""
Functions for evaluating algorithms and error
"""

import numpy as np
from nn_globals import NORM_FUNCTIONS
import net_prediction

def evaluate_error(predicted, actual, norm):
	assert(len(predicted) > 0)
	avg_levels = sum(actual) / float(len(actual))
	loss = NORM_FUNCTIONS[norm]
	diffs = [sum(loss(predicted[i], actual[i], avg_levels)) for i \
		in range(len(predicted))]
	return [diff / (predicted[0].shape[0]) for diff in diffs]

def evaluate_algorithm(scopes, algo, test_data_set, pollutant, norm, \
	hyper=None, model=None):
	future_scope = scopes[1]
	errors = np.zeros((future_scope,))
	count = 0.0
	for test_data in test_data_set:
		if algo == 'oracle' and len(test_data) < 2 * scopes[0] + future_scope + 2:
			continue
		count += 1
		actual_levels = net_prediction.isolate_pollutant_series(
			test_data, pollutant, scopes)
		if algo == 'feed-forward':
			predicted_levels = net_prediction.predict_next_nn_points(
				model, test_data, pollutant, hyper, False)
		elif algo == 'elman':
			predicted_levels = net_prediction.predict_next_nn_points(
				model, test_data, pollutant, hyper, True)
		elif algo == 'baseline':
			predicted_levels = net_prediction.predict_next_linear_points(
				test_data, pollutant, scopes)
		elif algo == 'oracle':
			predicted_levels = net_prediction.predict_middle_points(
				test_data, pollutant, scopes)
		err = evaluate_error(predicted_levels, actual_levels, norm)
		for j in range(future_scope):
			errors[j] += err[j] 
	errors /= count
	print 'Running Average Error'
	for i in range(len(errors)):
		print str(i + 1) + ': ' + str(sum(errors[: i + 1]) / float(i + 1))
