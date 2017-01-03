# Class to contain a row of pollution data
# Attributes include date, hour, average hourly levels of O3, NO2, CO, etc

import argparse
from datetime import datetime

class PollutionHour:
	def __init__(self, chron, station, pollutant_names, pollutant_vals, \
		weather_names, weather_vals):
		self.chron = datetime.strptime(chron, '%Y/%m/%d %H:%M')
		self.station = station
		self.pollutants = dict(zip(pollutant_names, pollutant_vals))
		self.weather = dict(zip(weather_names, weather_vals))

	def format_output(self):
		print 'Time: ' + self.chron.strftime('%Y/%m/%d %H:%M')
		print 'Station: ' + self.station
		for poll in sorted(self.pollutants):
			print poll + ': ' + self.pollutants[poll]
		for weath in sorted(self.weather):
			print weath + ': ' + self.weather[poll]




