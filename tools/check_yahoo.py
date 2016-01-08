#!/usr/bin/python2.7

""" Checks downloaded yahoo files, count by:
    - invalid status (eg, 'empty', 'bad_header', etc)
    - last trading date
"""

import os

RUN_ID = '20160101'
YAHOO_DIR = (
    '/Users/lnyang/lab/qd2/data/runs/%s/raw/yahoo/sf1' % RUN_ID)

def updateCount(stats, key):
  if key not in stats:
    stats[key] = 1
  else:
    stats[key] += 1

def printStats(stats):
  stats = [(k, v) for k, v in stats.items()]
  stats.sort(key=lambda item: item[1], reverse=True)
  for status, count in stats:
    print '  %s: %d' % (status, count)

yahoo_files = [f for f in os.listdir(YAHOO_DIR)
               if f.endswith('.csv')]
print 'detected %d csv files' % len(yahoo_files)

stats = dict()  # status => count
for f in yahoo_files:
  with open('%s/%s' % (YAHOO_DIR, f), 'r') as fp:
    lines = fp.read().splitlines()
  if len(lines) == 0:
    updateCount(stats, 'empty')
    continue
  if lines[0] != 'Date,Open,High,Low,Close,Volume,Adj Close':
    updateCount(stats, 'bad_header')
    continue
  if len(lines) == 1:
    updateCount(stats, 'no_data')
    continue
  da, op, hi, lo, cl, vo, ad = lines[1].split(',')
  updateCount(stats, da)

printStats(stats)

