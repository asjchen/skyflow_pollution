"""
Class to contain a row of pollution data
Attributes include date, hour, average hourly levels of O3, NO2, CO, etc
"""

from datetime import datetime

class PollutionHour:
	""" 
	Every instance of this class represents one hour of data 
	and contains all necessary information to characterize that hour
	"""

	def __init__(self, chron, station, pollutant_names, pollutant_vals, \
		weather_names, weather_vals):
		self.chron = datetime.strptime(chron, '%Y/%m/%d %H:%M')
		self.station = station
		self.pollutants = dict(zip(pollutant_names, pollutant_vals))
		self.weather = dict(zip(weather_names, weather_vals))

	def format_output(self):
		"""
		Capability for printing out a nicely formatted version of
		PollutionHour objects
		"""

		print 'Time: ' + self.chron.strftime('%Y/%m/%d %H:%M')
		print 'Station: ' + self.station
		for poll in sorted(self.pollutants):
			print poll + ': ' + self.pollutants[poll]
		for weath in sorted(self.weather):
			print weath + ': ' + self.weather[poll]

def get_pollutants(poll_hour):
    # accessor function returns list of pollutant values in poll_hour
    return [poll_hour.pollutants[x] for x in sorted(poll_hour.pollutants)]

def get_variables(poll_hour):
    # accessor returns list of variable values in poll_hour (pollutants first)
    weather_vars = [poll_hour.weather[x] for x in sorted(poll_hour.weather)]
    return get_pollutants(poll_hour) + weather_vars


