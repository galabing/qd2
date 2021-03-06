#!/usr/bin/python2.7

# Adapted from qd/train_model2.py
""" Trains a model based on training data, labels and model def.
    This is similar to train_model.py except that it takes all the data
    (see experiment Q) and selects a portion of it for training.  The
    selected portion is specified by --yyyymm and --months.
    Eg, with yyyymm=201012 and months=12, it will use all the
    data within [201001, 201012] for training.  It's the caller's
    responsibility that there is enough data within the specified
    period.

    For simplicity, it dumps selected portions of features and labels
    to temp files.  This can be improved.
"""

from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.tree import *
import argparse
import imputer_wrapper
import logging
import numpy
import os
import pickle
import util

MIN_SAMPLES = 10000

def selectData(data_file, label_file, meta_file, weight_file, train_meta_file,
               yyyymm, months, tmp_data_file, tmp_label_file, tmp_weight_file):
  assert len(yyyymm) == 6
  y = yyyymm[:4]
  m = yyyymm[4:]
  last_ym = '%s-%s' % (y, m)
  if months <= 0:
    first_ym = '0000-00'
  else:
    first_ym = util.getPreviousYm(last_ym, months - 1)
  logging.info('training period: %s - %s' % (first_ym, last_ym))
  assert first_ym <= last_ym

  data_ifp = open(data_file, 'r')
  data_ofp = open(tmp_data_file, 'w')
  label_ifp = open(label_file, 'r')
  label_ofp = open(tmp_label_file, 'w')
  if weight_file:
    weight_ifp = open(weight_file, 'r')
  if tmp_weight_file:
    weight_ofp = open(tmp_weight_file, 'w')

  meta_fp = open(meta_file, 'r')
  if train_meta_file is None:
    train_meta_fp = None
    train_meta = None
  else:
    train_meta_fp = open(train_meta_file, 'r')
    train_meta = train_meta_fp.readline()

  count = 0
  while True:
    meta = meta_fp.readline()
    if meta == '':
      assert data_ifp.readline() == ''
      assert label_ifp.readline() == ''
      if weight_file:
        assert weight_ifp.readline() == ''
      break
    data = data_ifp.readline()
    label = label_ifp.readline()
    assert data != ''
    assert label != ''
    if weight_file:
      weight = weight_ifp.readline()
      assert weight != ''

    if train_meta is not None:
      if meta != train_meta:
        continue
      train_meta = train_meta_fp.readline()

    assert meta[-1] == '\n'
    ticker, date, tmp1, tmp2 = meta[:-1].split('\t')
    ym = util.ymdToYm(date)
    if ym < first_ym or ym > last_ym:
      continue
    assert data[-1] == '\n'
    assert label[-1] == '\n'
    print >> data_ofp, data[:-1]
    print >> label_ofp, label[:-1]
    if tmp_weight_file:
      assert weight[-1] == '\n'
      print >> weight_ofp, weight[:-1]
    count += 1
    
  logging.info('selected %d training samples' % count)
  data_ifp.close()
  data_ofp.close()
  label_ifp.close()
  label_ofp.close()
  if weight_file:
    weight_ifp.close()
  if tmp_weight_file:
    weight_ofp.close()
  meta_fp.close()
  if train_meta_fp is not None:
    train_meta_fp.close()

def trainModel(data_file, label_file, weight_file, model_def, perc, imputer_strategy,
               model_file, imputer_file):
  X = numpy.loadtxt(data_file)
  y = numpy.loadtxt(label_file)
  if weight_file:
    w = numpy.loadtxt(weight_file)
  if X.shape[0] < MIN_SAMPLES:
    logging.info('too few samples: required %d, got %d' % (MIN_SAMPLES, X.shape[0]))
    return

  imputer = imputer_wrapper.ImputerWrapper(strategy=imputer_strategy)
  X = imputer.fit_transform(X)

  if perc < 1:
    logging.info('sampling %f data for training' % perc)
    m = int(X.shape[0] * perc)
    index = numpy.random.permutation(X.shape[0])[:m]
    X = X[index, :]
    y = y[index]
    if weight_file:
      w = w[index]

  model = eval(model_def)
  if weight_file:
    model.fit(X, y, w)
  else:
    model.fit(X, y)

  with open(model_file, 'wb') as fp:
    pickle.dump(model, fp)
  with open(imputer_file, 'wb') as fp:
    pickle.dump(imputer, fp)

def deleteTmpFiles(tmp_data_file, tmp_label_file):
  if os.path.isfile(tmp_data_file):
    os.remove(tmp_data_file)
  if os.path.isfile(tmp_label_file):
    os.remove(tmp_label_file)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_file', required=True)
  parser.add_argument('--label_file', required=True)
  parser.add_argument('--meta_file', required=True)
  # If specified, will be used in the fit function.
  parser.add_argument('--weight_file')
  # If specified, will be used to filter --meta_file.
  # Eg, --meta_file may contain metadata for all available data
  # while --train_meta_file may contain metadata for all data
  # with min_raw_price >= 10 and part of SP500 membership.
  # In this case, only data within --train_meta_file will be
  # collected for training, but --meta_file is still needed
  # for joining with --data_file and --label_file.
  parser.add_argument('--train_meta_file')
  parser.add_argument('--yyyymm', required=True,
                      help='last date of training period')
  parser.add_argument('--months', type=int, required=True,
                      help='length of training period in months, '
                           'use -1 to denote entire history')
  parser.add_argument('--model_def', required=True,
                      help='string of model def; eg, "Model(alpha=0.5)"')
  parser.add_argument('--perc', type=float, default=1.0,
                      help='if < 1, will randomly sample specified perc '
                           'of data for training')
  parser.add_argument('--imputer_strategy', default='zero',
                      help='strategy for filling in missing values')
  parser.add_argument('--model_file', required=True)
  parser.add_argument('--imputer_file', required=True)
  parser.add_argument('--tmp_data_file', required=True,
                      help='location of tmp data file within specified '
                           'training period; this can be used later for '
                           'evaluation, or specify --delete_tmp_files '
                           'to delete it upon finish')
  parser.add_argument('--tmp_label_file', required=True,
                      help='location of tmp label file within specified '
                           'training period; this can be used later for '
                           'evaluation, or specify --delete_tmp_files '
                           'to delete it upon finish')
  parser.add_argument('--tmp_weight_file')
  parser.add_argument('--delete_tmp_files', action='store_true')
  args = parser.parse_args()
  util.configLogging()
  if args.weight_file:
    assert args.tmp_weight_file, 'must specify --tmp_weight_file since --weight_file is specified'
  selectData(args.data_file, args.label_file, args.meta_file, args.weight_file,
             args.train_meta_file, args.yyyymm, args.months,
             args.tmp_data_file, args.tmp_label_file, args.tmp_weight_file)
  trainModel(args.tmp_data_file, args.tmp_label_file, args.tmp_weight_file,
             args.model_def, args.perc, args.imputer_strategy,
             args.model_file, args.imputer_file)
  if args.delete_tmp_files:
    deleteTmpFiles(args.tmp_data_file, args.tmp_label_file)
  # tmp_weight_file will not be used after this step so is not guarded by
  # --delete_tmp_files.
  if args.tmp_weight_file:
    os.remove(args.tmp_weight_file)

if __name__ == '__main__':
  main()

