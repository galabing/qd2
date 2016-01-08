#!/usr/bin/python2.7

# Adapted from qd.
""" Converts unzipped file of entire SF1 database into a set of raw files.

    Usage:
      ./convert_sf1_raw.py --sf1_file=SF1_20150502.csv
                           --indicator_file=./indicators.txt
                           --raw_dir=./raw

    It does one pass through sf1_file and separates lines of different
    tickers into different output raw files.  The raw files of each ticker
    should be small enough to be loaded into memory for further processing.
"""

import argparse
import datetime
import logging
import os
import util

# If this is modified readIndicatorMeta() also needs to be modified!
INDICATOR_HEADER = '\t'.join(['Indicator',
                              'Title',
                              'Available Dimensions',
                              'Statement',
                              'Description',
                              'NA Value',
                              'Units'])

# See https://www.quandl.com/data/SF1/documentation/indicators
# These indicators are either going through transitions or being
# populated.  None of them are used for training/prediction so
# we simply skip them here.
SKIPPED_INDICATORS = {
    'EVENT',
    'EVENTS',
    'RETEARN',
    'INVENTORY',
    'NCFCOMMON',
    'NCFDEBT',
    'NCFDIV',
}

def readIndicatorMeta(indicator_file):
  with open(indicator_file, 'r') as fp:
    lines = fp.read().splitlines()

  indicator_meta = dict()
  assert len(lines) > 0
  assert lines[0] == INDICATOR_HEADER
  for i in range(1, len(lines)):
    indicator, title, dimensions, statement, description, na, units = (
        lines[i].split('\t'))
    dimensions = [d for d in dimensions.split(',') if d != '']
    assert indicator not in indicator_meta
    indicator_meta[indicator] = set(dimensions)

  return indicator_meta

def processLine(line, indicator_meta):
  """ Processes one line from sf1_file, sanity-checks data, and returns
      ticker and line of output.  If skipped, returns ticker and reason.
  """
  # Eg, AA_ACCOCI_ARQ,2004-02-27,-569000000.0
  assert line[-1] == '\n'
  line = line[:-1]
  label, date, value = line.split(',')
  # Check label.
  items = label.split('_')
  assert len(items) == 2 or len(items) == 3
  ticker, indicator = items[0], items[1]
  if items[1] in SKIPPED_INDICATORS:
    return None, 'known_skipped'
  if indicator not in indicator_meta:
    return None, 'unknown_indicator'
  if len(items) == 2 and len(indicator_meta[indicator]) != 0:
    return None, 'expect_ND'
  if len(items) == 3 and items[2] not in indicator_meta[indicator]:
    return None, 'unknown_dimension'
  # Check date.
  try:
    tmp_date = datetime.datetime.strptime(date, '%Y-%m-%d')
  except ValueError:
    assert False, 'unsupported date format in line: %s' % line
  # Check value.
  try:
    tmp_value = float(value)
  except ValueError:
    assert False, 'unsupported value format in line: %s' % line

  return ticker, line

def convertSf1Raw(sf1_file, indicator_file, raw_dir, max_lines=0):
  # Since we always append to raw files (lines of a ticker may not be
  # adjacent in sf1_file) it's only sane to run this script with an
  # empty raw_dir.
  assert len(os.listdir(raw_dir)) == 0, (
      'nonempty raw dir: %s' % raw_dir)

  # Read indicator metadata for sanity-checking sf1_file.
  indicator_meta = readIndicatorMeta(indicator_file)

  num_lines = 0
  output_ticker = None
  output_fp = None

  stats = {
    'known_skipped': 0,
    'unknown_indicator': 0,
    'expect_ND': 0,
    'unknown_dimension': 0,
    'processed': 0,
  }

  with open(sf1_file, 'r') as fp:
    while True:
      line = fp.readline()
      if line == '':
        break
      # Prepare output data.
      ticker, line = processLine(line, indicator_meta)
      if ticker is None:
        stats[line] += 1
        continue
      stats['processed'] += 1
      # Prepare output fp.
      if ticker != output_ticker:
        if output_fp is not None:
          output_fp.close()
        output_fp = open('%s/%s' % (raw_dir, ticker), 'a')
        output_ticker = ticker
      print >> output_fp, line

      num_lines += 1
      if max_lines > 0 and num_lines >= max_lines:
        break
  if output_fp is not None:
    output_fp.close()

  logging.info('stats: %s' % stats)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--sf1_file', required=True,
                      help='unzipped file of entire SF! database from quandl')
  parser.add_argument('--indicator_file', required=True,
                      help='file of supported indicators in SF1')
  parser.add_argument('--raw_dir', required=True,
                      help='output dir of raw files')
  parser.add_argument('--max_lines', type=int, default=0,
                      help='max number of lines to process from sf1_file; '
                           'only use this for debugging')
  args = parser.parse_args()
  util.configLogging()
  convertSf1Raw(args.sf1_file, args.indicator_file, args.raw_dir,
                args.max_lines)

if __name__ == '__main__':
  main()

