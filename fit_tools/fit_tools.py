import sys
from collections import namedtuple
from collections import defaultdict

from fitparse import FitFile, FitParseError
import pandas as pd
import numpy as np

FitWindow = namedtuple('FitWindow', ['start', 'end'])

class FitInterval:
  def __init__(self, df, start, end):
    self.df = df
    self.start = start.to_timestamp()
    self.end = end.to_timestamp()

  def to_dataframe(self):
    return self.df[self.start:self.end]

  def overlaps(self, interval):
    if interval.end < self.start or interval.start > self.end:
      return False
    return True

def ParseIntervalSpec(spec):
  """ The interval spec is a comma separated list of count@time
      pairs. Time can be specified in minutes or seconds using 'm' or 's'
      respectivly.
      Example:
       2@30s,2@15m
      Returns list of tuple: [(count, time in seconds)]
"""
  def ParseSpec(spec):
    fields = spec.split('@')
    if len(fields) != 2:
      raise Exception('Expected 2 fields in interval spec: ' + spec)
    count = int(fields[0])
    duration = fields[1]
    # last char is unit (s or m)
    unit = duration[-1]
    val = int(duration[:-1])
    if unit == 'm':
      val *= 60
    elif unit != 's':
      raise Exception('Unknown duration unit in interval spec: ' + unit)
    return int(count), val

  specs = spec.split(',')
  return [ParseSpec(spec) for spec in specs]

def FitToDataframe(fitfile, fields):
  """
  Function takes a fit file object and a list of fields to extract.
  All fields will be read from the fit record and converted to a
  series in the returned DataFrame.
  The timestamp field is always extracted and should not be specified

  Arguments: fitfile - parsed FitFile object
             fields - list of field names to include as columns.
  Returns:   Pandas DataFrame object.
  """
  # Get all data messages that are of type record
  msgs = fitfile.get_messages('record')
  records = [record for record in msgs]
  data = defaultdict(list)

  # fields always includes timestamp:
  user_fields = fields[:]
  fields = ['timestamp'] + user_fields

  # Read each requested field into a list, keyed by field name.
  for record in records:
    for field in fields:
      val = record.get_value(field)
      data[field].append(record.get_value(field))

  # Create time series dataframe in pandas
  df = pd.DataFrame(data, columns = fields)

  # Interpolate to fill in missing values.
  df[user_fields] = df[user_fields].interpolate(method='cubic')

  # Change datatype of timestamp field
  df['timestamp'] = pd.to_datetime(df['timestamp'])

  # Compute elapsed time column
  elapsed_seconds = len(df['timestamp'])

  df['elapsed'] = pd.date_range('00:00:00', periods=elapsed_seconds, freq='S')
  return df.set_index('timestamp')

def FitPowerCurve(fit):
  # TODO(Cheradenine): not working yet.
  df = FitToDataframe(fit, ['power'])[['power']]
  seconds = len(df.index)

  def MaxWindow(df, size):
    wdf = df.rolling(window=size).aggregate(np.mean)
    return wdf['power'].max()
  curve = [MaxWindow(df, size+1) for size in range(0, seconds)]
  return pd.DataFrame(curve, index = df.index)

class FitIntervals:
  """
  Builds a list of non-overlapping power intervals.
  """

  def __init__(self, fit):
    self.df = FitToDataframe(fit, ['power', 'heart_rate'])
    self.intervals = []

  def FindIntervals(self, length, count):
    """ Find a set of intervals.
    Arguments: length - lengh  of interval in seconds
               count - number of intervals to find.
    Returns: None
    """
    df = self.df[['power']]
    df = df.rolling(window=length).aggregate(np.mean)
    df = df.sort_values(by='power', ascending=0)
    max_count = len(self.intervals) + count
    for index, row in df.iterrows():
      power = row['power']
      period = pd.Period(index, freq='S')
      # Windows start on the right edge by default. We want the left edge.
      interval = FitInterval(self.df, start=period-length, end=period)
      # determine if this new interval overlaps with any seen so far
      if not any(interval.overlaps(i) for i in self.intervals):
        self.intervals.append(interval)
        if len(self.intervals) == max_count:
          break
    # Sort intervals by start time.
    self.intervals.sort(key=lambda i:i.start)

  def Report(self):
    # TODO(Cheradenine): clean this up.
    for interval in self.intervals:
      # Slice the data around the interval.
      df = interval.to_dataframe()
      print('time={} - {}, power={:03.0f}, hr={:03.0f}'.format(
          interval.start, interval.end, df['power'].mean(), df['heart_rate'].mean()))

def ComputePowerCurve(fname):
  fit = FitFile(fname)
  df = FitPowerCurve(fit)
  return df

if __name__ == "__main__":
  """ Implementes a command line tool to find intervals.
  Usage: fit_tools 5@5m,5@30s <fit_file>
  This would find 5 intervals of 5 mins and 5 intervals of 30 secods.
  """
  # TODO(Cheradenine): better command line parsing, error reporting.
  specs = ParseIntervalSpec(sys.argv[1])
  try:
    fit = FitFile(sys.argv[2])
    fit.parse()
  except FitParseError as e:
    print('Unable to parse fit file: {}'.format(e))
    sys.exit(1)
  intervals = FitIntervals(fit)
  for count, duration in specs:
    intervals.FindIntervals(duration, count)
  intervals.Report()
