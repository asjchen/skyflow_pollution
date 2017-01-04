"""
Oracle algorithm uses quadratic regression with knowledge of the points
surrounding the point we want to predict. In regular circumstances, we would 
not be able to have this information; the purpose of this algorithm is to 
provide an upper bound on the expected performance of the actual neural 
network. 
"""

import data_util
import test_util
import numpy as np

def evaluate_oracle(pollution_dir_test, radius, future_scope, pollutant, norm):
	test_data_set = data_util.data_from_directory(pollution_dir_test)
	scopes = (radius, future_scope)
	test_util.evaluate_algorithm(scopes, 'oracle', test_data_set, \
		pollutant, norm)

def parse_oracle_input(args):
	args.pollution_dir_test = data_util.remove_slash(args.pollution_dir_test)
	evaluate_oracle(args.pollution_dir_test, args.radius, args.future_scope, \
		args.chemical, args.norm)

