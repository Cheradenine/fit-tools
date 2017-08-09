import sys
from collections import namedtuple
from collections import defaultdict

from fitparse import FitFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

FitWindow = namedtuple('FitWindow', ['start', 'end'])

# Returns True if the window overlaps with a list of existing  windows
def WindowOverlaps(windows, window):
  for w in windows:
    if window.start >= w.start and window.start <= w.end:
      return True
    if window.end >= w.start and window.end <= w.end:
      return True
    if window.start <= w.start and window.end >= w.end:
      return True
  return False

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
  fields = ['timestamp'] + fields

  # Read each requested field into a list, keyed by field name.
  for record in records:
    for field in fields:
      data[field].append(record.get_value(field))

  # Create time series dataframe in pandas
  df = pd.DataFrame(data, columns = fields)

  # Change datatype of timestamp field
  df['timestamp'] = pd.to_datetime(df['timestamp'])
  return df.set_index('timestamp')

class FitIntervals:
  """
  Contains a list of non-overlapping power intervals.
  """

  def __init__(self, file_name):
    self.fit = FitFile(file_name)
    self.df = FitToDataframe(self.fit, ['power', 'heart_rate'])
    self.intervals = []

  def FindIntervals(self, size, count):
    df = self.df.rolling(window=size).aggregate(np.mean)
    df = df.sort_values(by='power', ascending=0)
    max_count = len(self.intervals) + count
    for index, row in df.iterrows():
      power = row['power']
      period = pd.Period(index, freq='S')
      # Windows start on the right edge by default.
      window = FitWindow(start=period-size, end=period)
      if not WindowOverlaps(self.intervals, window):
        self.intervals.append(window)
        if len(self.intervals) == max_count:
          break

  def Report(self):
    # TODO(Cheradine): clean this up.
    for interval in self.intervals:
      # Slice the data around the interval.
      df = self.df[interval.start.to_timestamp():interval.end.to_timestamp()]
      print('time={} - {}, power={:03.0f}, hr={:03.0f}'.format(
          interval.start, interval.end, df['power'].mean(), df['heart_rate'].mean()))


if __name__ == "__main__":
  specs = ParseIntervalSpec(sys.argv[1])
  fname = sys.argv[2]
  intervals = FitIntervals(fname)
  for count, duration in specs:
    intervals.FindIntervals(duration, count)
  intervals.Report()
