# Functions for testing

import numpy as np
from nn_globals import *

def evaluate_error(predicted, actual, norm):
	assert(len(predicted) > 0)
	avg_levels = sum(actual) / float(len(actual))
	diffs = [sum(NORM_FUNCTIONS[norm](predicted[i], actual[i], avg_levels)) for i \
		in range(len(predicted))]
	return [diff / (predicted[0].shape[0]) for diff in diffs]

