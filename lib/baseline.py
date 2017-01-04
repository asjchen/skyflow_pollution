"""
Baseline algorithm to predict the levels of pollutants in the next few hours.
The purpose of this simple algorithm is to provide a lower bound on the 
expected performance of the actual neural network. Here, we run linear 
regression on the last [past_scope] hours for each pollutant.
"""

import input_util, test_util, net_prediction
import random
import numpy as np

def main():
	input_args = input_util.parse_baseline_input()
	pollution_dir_test, past_scope, future_scope, pollutant, norm = input_args
	test_data_set = input_util.data_from_directory(pollution_dir_test)
	errors = np.zeros((future_scope,))
	scopes = (past_scope, future_scope)
	for test_data in test_data_set:
		predicted_levels = net_prediction.predict_next_linear_points(
			test_data, pollutant, scopes)
		actual_levels = net_prediction.isolate_pollutant_series(
			test_data, pollutant, scopes)
		err = test_util.evaluate_error(predicted_levels, actual_levels, norm)
		for j in range(future_scope):
			errors[j] += err[j]
	errors /= float(len(test_data_set))
	print 'Running Average Error'
	for i in range(len(errors)):
		print str(i + 1) + ': ' + str(sum(errors[: i + 1]) / float(i + 1))

if __name__ == '__main__':
	main()
