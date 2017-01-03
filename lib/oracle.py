# Oracle algorithm uses quadratic regression with knowledge of the 5 points
# surrounding the point we want to predict

import input_util, net_prediction, test_util
import numpy as np
import scipy.stats

def main():
	pollution_dir_test, radius, future_scope, pollutant, norm = input_util.parse_oracle_input()
	test_data_set = input_util.data_from_directory(pollution_dir_test)
	errors = np.zeros((future_scope,))
	count = 0.0
	for test_data in test_data_set:
		if len(test_data) < 2 * radius + future_scope + 2:
			continue
		count += 1
		predicted_levels = net_prediction.predict_middle_points(test_data, pollutant, radius, future_scope)
		actual_levels = net_prediction.isolate_pollutant_series(test_data, pollutant, radius, future_scope)
		err = test_util.evaluate_error(predicted_levels, actual_levels, norm)
		for j in range(future_scope):
			errors[j] += err[j] 
	errors = errors / count
	print 'Running Average Error'
	for i in range(len(errors)):
		print str(i + 1) + ': ' + str(sum(errors[: i + 1]) / float(i + 1))

if __name__ == '__main__':
	main()
