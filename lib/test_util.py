"""
Functions for evaluating algorithms and error
"""

import numpy as np
from nn_globals import NORM_FUNCTIONS
import net_prediction

def evaluate_error(predicted, actual, norm):
	""" Evaluates the error on a certain prediction

	predicted:		predicted pollutant values (format: list of lists of pollutant values)
	actual:			actual pollutant values (format: list of lists of pollutant values)
	norm:			name of the appropriate norm for evalution

	Returns the error vector for the corresponding pollutants
	"""
	assert(len(predicted) > 0)
	avg_levels = sum(actual) / float(len(actual))
	loss = NORM_FUNCTIONS[norm]
	diffs = [sum(loss(predicted[i], actual[i], avg_levels)) for i \
		in xrange(len(predicted))]
	return [diff / (predicted[0].shape[0]) for diff in diffs]

def evaluate_algorithm(scopes, algo, test_data_set, pollutant, norm, \
	hyper=None, model=None):
	""" Evaluates a given pollutant prediction algorithm and prints error vals

	scopes:			tuple of form (past_scope, future_scope) where
					past_scope is the number of past data points to 
					use for prediction and future scope is the number
					of hours in the future to predict 
	algo:			name of the algorithm to test -- choose from 
					['oracle', 'baseline', 'feed-forward', 'elman']
	test_data_set:	Data set on which to test (format: list of list of PollutionHours)
					see data_util.py for more understanding of format
	pollutant:		name of pollutant to graph output for
	norm:			name of norm to use for prediction
	hyper:			object containing hyperparameter data (NN only)
	model:			object containing model data (NN only)
	"""

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
		for j in xrange(future_scope):
			errors[j] += err[j] 
	errors /= count
	print 'Running Average Error'
	for i in xrange(len(errors)):
		print str(i + 1) + ': ' + str(sum(errors[: i + 1]) / float(i + 1))
