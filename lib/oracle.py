"""
Oracle algorithm uses quadratic regression with knowledge of the points
surrounding the point we want to predict. In regular circumstances, we would 
not be able to have this information; the purpose of this algorithm is to 
provide an upper bound on the expected performance of the actual neural 
network. 
"""

import input_util, net_prediction, test_util
import numpy as np

def main():
	input_args = input_util.parse_oracle_input()
	pollution_dir_test, radius, future_scope, pollutant, norm = input_args
	test_data_set = input_util.data_from_directory(pollution_dir_test)
	scopes = (radius, future_scope)
	test_util.evaluate_algorithm(scopes, 'oracle', test_data_set, \
		pollutant, norm)

if __name__ == '__main__':
	main()
