#!/usr/bin/python2.7

""" Collects label/weight for training data.

    Example usage:
      ./collect_labels.py --meta_file=./meta
                          --min_pos=0
                          --max_neg=0
                          --label_file=./label
                          --weight_file=./weight
      or
      ./collect_labels.py --meta_file=./meta
                          --min_pos_perc=0.75
                          --max_neg_perc=0.75
                          --label_file=./label
                          --weight_file=./weight
    If min_pos/max_neg are specified, training data is classified into
      positive iff gain > min_pos,
      negative iff gain < max_neg.
    If min_pos_perc/max_neg_perc are specified, training data is grouped
    by date and sorted by ascending gain.  Highest min_pos_perc percentile
    are classified as positive and lowest max_neg_perc percentile are
    classified as negative.

    In both cases there should be no overlap between min_pos(_perc) and
    max_neg(_perc).

    Optional: weight_file stores abs(gain - threshold) for each data point.
    If --weight_power is set weights are raised to specified power.
    Default weight_power is 1.
"""

import argparse
import logging
import util

def check_args(args):
  use_raw = (args.min_pos is not None and args.max_neg is not None)
  use_perc = (args.min_pos_perc is not None and args.max_neg_perc is not None)
  if use_raw:
    assert not use_perc
    assert args.min_pos >= args.max_neg
  if use_perc:
    assert not use_raw
    assert args.min_pos_perc < 1
    assert args.max_neg_perc > 0
    assert args.min_pos_perc >= args.max_neg_perc
  return use_perc

def read_meta(meta_file):
  with open(meta_file, 'r') as fp:
    lines = fp.read().splitlines()
  meta = []
  for line in lines:
    _, date, _, gain = line.split('\t')
    meta.append([date, float(gain)])
  return meta

def get_label_weight(gain, min_pos, max_neg, weight_power):
  if gain > min_pos:
    label = 1
    weight = gain - min_pos
  elif gain < max_neg:
    label = 0
    weight = max_neg - gain
  else:
    label = -1
    weight = 0
  if weight_power != 1:
    weight = weight**weight_power
  return label, weight

def collect_raw(meta, min_pos, max_neg, weight_power):
  labels, weights = [], []
  for _, gain in meta:
    label, weight = get_label_weight(gain, min_pos, max_neg, weight_power)
    labels.append(label)
    weights.append(weight)
  return labels, weights

def collect_perc(meta, min_pos_perc, max_neg_perc, weight_power):
  dgains = dict()  # date => [gain]
  for date, gain in meta:
    if date not in dgains:
      dgains[date] = [gain]
    else:
      dgains[date].append(gain)
  for date in dgains:
    dgains[date].sort()

  dthresh = dict()  # date => (min_pos, max_neg)
  for date, gains in dgains.iteritems():
    min_pos = gains[int(len(gains)*min_pos_perc)]
    max_neg = gains[int(len(gains)*max_neg_perc)]
    dthresh[date] = (min_pos, max_neg)

  labels, weights = [], []
  for date, gain in meta:
    min_pos, max_neg = dthresh[date]
    label, weight = get_label_weight(gain, min_pos, max_neg, weight_power)
    labels.append(label)
    weights.append(weight)
  return labels, weights

def collect(args):
  use_perc = check_args(args)
  meta = read_meta(args.meta_file)
  if use_perc:
    labels, weights, = collect_perc(
        meta, args.min_pos_perc, args.max_neg_perc, args.weight_power)
  else:
    labels, weights, = collect_raw(
        meta, args.min_pos, args.max_neg, args.weight_power)

  counts = {1: 0, 0: 0, -1: 0}
  for label in labels:
    counts[label] += 1
  logging.info('label counts: %s' % counts)
  
  with open(args.label_file, 'w') as fp:
    for label in labels:
      print >> fp, '%f' % label
  if args.weight_file is not None:
    with open(args.weight_file, 'w') as fp:
      for weight in weights:
        print >> fp, '%f' % weight

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--meta_file', required=True)
  parser.add_argument('--min_pos', type=float)
  parser.add_argument('--max_neg', type=float)
  parser.add_argument('--min_pos_perc', type=float)
  parser.add_argument('--max_neg_perc', type=float)
  parser.add_argument('--label_file', required=True)
  parser.add_argument('--weight_power', type=float, default=1)
  parser.add_argument('--weight_file')
  util.configLogging()
  collect(parser.parse_args())

if __name__ == '__main__':
  main()

