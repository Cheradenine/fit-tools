import sys
from collections import namedtuple
from collections import defaultdict

from fitparse import FitFile
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
      interval = FitInterval(self.df, start=period-size, end=period)
      # determine if this new interval overlaps with any seen so far
      if not any(interval.overlaps(i) for i in self.intervals):
        self.intervals.append(interval)
        if len(self.intervals) == max_count:
          break

  def Report(self):
    # TODO(Cheradine): clean this up.
    for interval in self.intervals:
      # Slice the data around the interval.
      df = interval.to_dataframe()
      print('time={} - {}, power={:03.0f}, hr={:03.0f}'.format(
          interval.start, interval.end, df['power'].mean(), df['heart_rate'].mean()))


if __name__ == "__main__":
  size = int(sys.argv[1])
  count = int(sys.argv[2])
  fname = sys.argv[3]
  intervals = FitIntervals(fname)
  intervals.FindIntervals(120, count)
  intervals.FindIntervals(size, count)
  intervals.Report()
