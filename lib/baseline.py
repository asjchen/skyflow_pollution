"""
Baseline algorithm to predict the levels of pollutants in the next few hours.
The purpose of this simple algorithm is to provide a lower bound on the 
expected performance of the actual neural network. Here, we run linear 
regression on the last [past_scope] hours for each pollutant.
"""

import data_util
import test_util
import numpy as np

def evaluate_baseline(pollution_dir_test, past_scope, future_scope, pollutant, norm):
	test_data_set = data_util.data_from_directory(pollution_dir_test)
	scopes = (past_scope, future_scope)
	test_util.evaluate_algorithm(scopes, 'baseline', test_data_set, \
		pollutant, norm)

def parse_baseline_input(args):
	args.pollution_dir_test = data_util.remove_slash(args.pollution_dir_test)
	evaluate_baseline(args.pollution_dir_test, args.past_scope, \
		args.future_scope, args.chemical, args.norm)

