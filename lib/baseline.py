"""
Baseline algorithm to predict the levels of pollutants in the next few hours.
The purpose of this simple algorithm is to provide a lower bound on the 
expected performance of the actual neural network. Here, we run linear 
regression on the last [past_scope] hours for each pollutant.
"""

import input_util, test_util
import random
import numpy as np

def main():
	input_args = input_util.parse_baseline_input()
	pollution_dir_test, past_scope, future_scope, pollutant, norm = input_args
	test_data_set = input_util.data_from_directory(pollution_dir_test)
	scopes = (past_scope, future_scope)
	test_util.evaluate_algorithm(scopes, 'baseline', test_data_set, \
		pollutant, norm)

if __name__ == '__main__':
	main()
