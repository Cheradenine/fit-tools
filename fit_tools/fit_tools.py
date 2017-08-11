import sys
from collections import namedtuple
from collections import defaultdict

from fitparse import FitFile, FitParseError
import pandas as pd
import numpy as np

"""
Tuple to hold interval information:
start = the start timestamp of the interval
end = the end timestamp of the interval
"""
FitInterval = namedtuple('FitInterval', ['start', 'end'])

def IntervalsOverlap(i1, i2):
  """ Determines if two FitIntervals overlap.
  Returns - True if the intervals overlap, False otherwise
 """
  if i1.end < i2.start or i1.start > i2.end:
    return False
  return True

def ParseIntervalSpec(specs):
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

  specs = specs.split(',')
  return [ParseSpec(spec) for spec in specs]

def FitToDataFrame(fitfile, fields, interpolate = True):
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
  if interpolate:
    df[user_fields] = df[user_fields].interpolate(method='cubic')

  # Change datatype of timestamp field
  df['timestamp'] = pd.to_datetime(df['timestamp'])

  # Compute elapsed time column
  elapsed_seconds = len(df['timestamp'])

  df['elapsed'] = pd.date_range('00:00:00', periods=elapsed_seconds, freq='S')
  return df.set_index('timestamp')

def FitPowerCurve(fit):
  # TODO(Cheradenine): not working yet.
  df = FitToDataFrame(fit, ['power'])[['power']]
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
    self.df = FitToDataFrame(fit, ['power', 'heart_rate'])
    self.intervals = []

  def __iter__(self):
    """
    Returns - iterable of FitInterval
    """
    return iter(self.intervals)

  def Find(self, length, count):
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
      start = (period-length).to_timestamp()
      end = period.to_timestamp()
      # Windows start on the right edge by default. We want the left edge.
      interval = FitInterval(start=start, end=end)
      # determine if this new interval overlaps with any seen so far
      if not any(IntervalsOverlap(interval, i) for i in self.intervals):
        self.intervals.append(interval)
        if len(self.intervals) == max_count:
          break
    # Sort intervals by start time.
    self.intervals.sort(key=lambda i:i.start)

  def IntervalData(self, interval):
    return self.df[interval.start:interval.end]

  def Report(self):
    # TODO(Cheradenine): clean this up.
    for interval in self.intervals:
      # Slice the data around the interval.
      df = self.df[interval.start:interval.end]
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
    intervals.Find(duration, count)
  intervals.Report()
