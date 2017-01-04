# Input Utilities

import sys
import os
import argparse
import csv
import re
from pollution_hour import PollutionHour
from nn_globals import NetHyperparams
from nn_globals import NORM_FUNCTIONS, NORM_GRADIENTS
from nn_globals import ACTIVATION_FUNCTIONS, ACTIVATION_GRADIENTS

default_past_scope = 12
default_future_scope = 12
default_hidden_dim = 100
default_activation = 'tanh'
default_num_iterations = 20
default_norm = 'L1'
default_step_scale = 0.05

pollutant_names = ['CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'SO2']
weather_names = ['AMB_TEMP', 'RAINFALL', 'RH', 'WD_HR', 'WIND_DIREC', \
	'WIND_SPEED', 'WS_HR']
	
def parse_taiwanese_csv(csv_file, include_invalid = False, \
	include_headers = False):
	reader = list(csv.reader(csv_file))
	reader = [line for line in reader if len(line) > 1]
	csv_file.close()
	# specific to the Taiwanese CSV data file
	global pollutant_names
	global weather_names
	headers = [name for name \
		in reader[0] if name in pollutant_names or name in weather_names]
	headers = ['time', 'station'] + headers
	pollution_data = []
	row_idx = dict([(name, reader[0].index(name)) for name \
		in pollutant_names + weather_names])
	for i in range(1, len(reader)):
		if i % 1000 == 0:
			print i
		chron = reader[i][0]
		station = reader[i][1]
		non_decimal = re.compile(r'[^\d.-]+')
		pollutant_vals = []
		no_rain = {'RAINFALL': 0.0, 'PH_RAIN': 7.0, 'RAIN_COND': 0.0}
		has_invalid = False
		for name in pollutant_names:
			value = non_decimal.sub('', reader[i][row_idx[name]])
			val_str = reader[i][row_idx[name]].strip()
			if value != val_str or val_str == '':
				if name in no_rain and (val_str == 'NR' or val_str == ''):
					pollutant_vals.append(no_rain[name])
				else:
					has_invalid = True
					pollutant_vals.append(None)
			else:
				pollutant_vals.append(float(value))

		weather_vals = []
		for name in weather_names:
			value = non_decimal.sub('', reader[i][row_idx[name]])
			val_str = reader[i][row_idx[name]].strip()
			if value != val_str or val_str == '':
				if name in no_rain and (val_str == 'NR' or val_str == ''):
					weather_vals.append(no_rain[name])
				else:
					has_invalid = True
					weather_vals.append(None)
			else:
				weather_vals.append(float(value))
		if not has_invalid or include_invalid:
			pollution_data.append(PollutionHour(chron, station, \
				pollutant_names, pollutant_vals, weather_names, weather_vals))
	if not include_headers:
		return pollution_data
	else:
		return pollution_data, headers

def data_from_directory(pollution_dir):
	pollution_data_list = []
	for dirpath, dirnames, filenames in os.walk(pollution_dir):
		for f in filenames:
			if f[0] == '.': continue
			pollution_csv = open(dirpath + '/' + f, 'r')
			pollution_data_list.append(parse_taiwanese_csv(pollution_csv))
			pollution_csv.close()
	return pollution_data_list

############## FUNCTIONS FOR PARSING COMMAND LINE INPUT ####################

def remove_slash(name):
	if name[-1] == '/':
		return name[: -1]
	return name

def make_base_parser(parser):
	parser.add_argument('-f', '--future_scope', type=int, \
		default=default_future_scope, \
		help='Scope of future -- number of points to predict')
	parser.add_argument('-c', '--chemical', default=None, \
		help='Pollutant to graph/output time series for')
	parser.add_argument('-n', '--norm', default=default_norm, \
		choices=NORM_FUNCTIONS.keys(), \
		help='Loss function, choose from ' + str(NORM_FUNCTIONS.keys()))

def parse_baseline_input():
	baseline_descr = ('Baseline algorithm -- batch linear regression'
		' on pollution levels')
	parser = argparse.ArgumentParser(description=baseline_descr)
	make_base_parser(parser)
	parser.add_argument('pollution_dir_test', type=str, \
		help='Directory with the test pollution data')
	parser.add_argument('-p', '--past_scope', type=int, \
		default=default_past_scope, \
		help='Scope of past -- number of points to interpolate from')
	args = parser.parse_args()
	args.pollution_dir_test = remove_slash(args.pollution_dir_test)
	return (args.pollution_dir_test, args.past_scope, args.future_scope, \
		args.chemical, args.norm)

def parse_oracle_input():
	oracle_descr = ('Oracle algorithm -- localized quadratic regression at '
		'one point at a time, with data from both before and after '
		'each target point')
	parser = argparse.ArgumentParser(description=oracle_descr)
	make_base_parser(parser)
	parser.add_argument('pollution_dir_test', type=str, \
		help='Directory with the test pollution data')
	parser.add_argument('-r', '--radius', type=int, \
		default=default_past_scope, \
		help='Radius around the target data point to interpolate from')
	args = parser.parse_args()
	args.pollution_dir_test = remove_slash(args.pollution_dir_test)
	return (args.pollution_dir_test, args.radius, args.future_scope, \
		args.chemical, args.norm)

def parse_nn_input():
	nn_descr = 'Neural Network -- runs on a sliding window of the input data'
	top_parser = argparse.ArgumentParser(description=nn_descr)
	subparsers = top_parser.add_subparsers(title='subcommands',
		description='valid subcommands', help='Neural network types')
	forward_parser = subparsers.add_parser('feed-forward', 
		help='Feed-Forward NN')
	forward_parser.set_defaults(has_feedback=False)
	elman_parser = subparsers.add_parser('elman', help='Elman RNN')
	elman_parser.set_defaults(has_feedback=True)
	subparser_list = [forward_parser, elman_parser]
	for i in range(2):
		parser = subparser_list[i]
		make_base_parser(parser)
		parser.add_argument('pollution_dir_train', type=str, \
			help='Directory with the train pollution data')
		parser.add_argument('pollution_dir_test', type=str, \
			help='Directory with the test pollution data')
		parser.add_argument('-d', '--hidden_dim', type=int, \
			default=default_hidden_dim, \
			help='Dimension of the hidden layer in a 3-layer NN')
		activ_help = 'Activation function for the middle layer, choose from '
		parser.add_argument('-a', '--activation', type=str, \
			default=default_activation, \
			choices = ACTIVATION_FUNCTIONS.keys(), \
			help=activ_help + str(ACTIVATION_FUNCTIONS.keys()))
		parser.add_argument('-p', '--past_scope', type=int, \
			default=default_past_scope, \
			help='Number of hours in for each iteration of the sliding window')
		parser.add_argument('--reg_w1', type=float, default=0.0, 
			help='Regularization parameter for W1')
		parser.add_argument('--reg_b1', type=float, default=0.0, 
			help='Regularization parameter for b1')
		parser.add_argument('--reg_w2', type=float, default=0.0, 
			help='Regularization parameter for W2')
		parser.add_argument('--reg_b2', type=float, default=0.0, 
			help='Regularization parameter for b2')
		parser.add_argument('--reg_u', type=float, default=0.0, 
			help='Regularization parameter for U')
		parser.add_argument('-i', '--num_iterations', type=int, \
			default=default_num_iterations, \
			help='Number of iterations/passes over the training data')
		parser.add_argument('-s', '--step_scale', type=float, \
			default=default_step_scale, \
			help='The value with which to scale the step size')
	args = top_parser.parse_args()
	args.pollution_dir_test = remove_slash(args.pollution_dir_test)
	args.pollution_dir_train = remove_slash(args.pollution_dir_train)
	pollution_dirs = (args.pollution_dir_train, args.pollution_dir_test)
	reg_params = { 'W1': args.reg_w1, 'b1': args.reg_b1, 'W2': args.reg_w2, \
		'b2': args.reg_b2, 'U': args.reg_u }
	hyper = NetHyperparams(args.hidden_dim, args.activation, args.past_scope, \
		reg_params, args.num_iterations, args.future_scope, args.norm, \
		args.step_scale)
	return (args.has_feedback, pollution_dirs, hyper, args.chemical)

