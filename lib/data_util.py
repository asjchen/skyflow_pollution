import os
import csv
import re
from pollution_hour import PollutionHour

pollutant_names = ['CO', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5', 'SO2']
weather_names = ['AMB_TEMP', 'RAINFALL', 'RH', 'WD_HR', 'WIND_DIREC', \
	'WIND_SPEED', 'WS_HR']

def remove_slash(name):
	if name[-1] == '/':
		return name[: -1]
	return name	

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