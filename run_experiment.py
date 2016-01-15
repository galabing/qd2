#!/usr/bin/python2.7

""" Runs one experiment.
    Example usage:
      ./run_experiment.py --config=./config.json

    The config specifies:
    - features (eg, 'sf1 + eod + eod-R3000 + sector')
    - label (eg, 'yahoo + R3000')
    - start_date (eg, '200412')
    - end_date (eg, '201305')
    - model_spec (default
      'RandomForestClassifier(n_estimators=100, max_depth=2)')
    - feature_window (defines max lag in days in joining features, default 120)
    - train_window (defines training period in months, default -1 using entire
      history)
    - train_perc (default 1.0)
    - imputer_strategy (default 'zero', can also be 'mean', 'median')
    - predict_window (defines prediction window in months, default 12)
    - delay_window (defines delay in months in model training, default 0 no
      delay)
    - min_date (default '0000-00-00')
    - max_date (default '9999-99-99')
    - train_filter (default '')
    - predict_filter (default '')
    - max_neg (default 0)
    - min_pos (default 0)
    - use_perc (treat max_neg/min_pos and max_neg_perc/min_pos_perc,
      default False)
    - use_weight (default False)
    - weight_power (default 1)
    - use_classification (default True), set to False to use regression

    TODO: start_date and end_date are hand picked for now but can be automated.
    TODO: use_classification must be True after the addition of collect_labels.py.
          Add back rlabel output if regression is desired.
"""

from config import *
from sklearn.metrics import *
import argparse
import config
import json
import numpy
import os
import pickle
import util

CONFIG_SUFFIX = '.json'
TMP_DATA_FILE = '/tmp/qd2_tmp_data'
TMP_LABEL_FILE = '/tmp/qd2_tmp_label'
TMP_WEIGHT_FILE = '/tmp/qd2_tmp_weight'

REQUIRED_FIELDS = [
    'features',
    'label',
    'start_date',
    'end_date',
]
DEFAULT_VALUES = {
    'model_spec': 'RandomForestClassifier(n_estimators=100, max_depth=2)',
    'feature_window': 120,
    'train_window': -1,
    'train_perc': 1.0,
    'imputer_strategy': 'zero',
    'predict_window': 12,
    'delay_window': 0,
    'max_neg': 0.0,
    'min_pos': 0.0,
    'use_perc': False,
    'min_date': '0000-00-00',
    'max_date': '9999-99-99',
    'min_feature_perc': 0.8,
    'train_filter': '',
    'predict_filter': '',
    'use_weight': False,
    'weight_power': 1.0,
    'use_classification': True,
}

def getConfig(config_file):
  with open(config_file, 'r') as fp:
    config_map = json.load(fp)
  for field in REQUIRED_FIELDS:
    assert field in config_map, 'missing field %s in config' % field
  for field, value in DEFAULT_VALUES.iteritems():
    if field not in config_map:
      config_map[field] = value
  return config_map

def getFeatureListPath(experiment_dir):
  return '%s/feature_list' % experiment_dir

def getSourceAndMarket(label):
  source, market = [item.strip() for item in label.split('+')]
  assert source in ['eod', 'yahoo'], 'unknown label in config: %s' % label
  assert market in MARKETS, 'unknown label in config: %s' % label
  return source, market

# Dir of label SOURCE data (egain dir).
# This is not the dir to training labels (getLabelPath()).
def getLabelDir(label):
  source, market = getSourceAndMarket(label)
  if source == 'eod':
    return '%s/%s' % (EOD_EGAIN_LABEL_DIR, market)
  return '%s/%s' % (YAHOO_EGAIN_LABEL_DIR, market)

# Dir of market gains.
def getMarketGainPath(config_map):
  _, market = getSourceAndMarket(config_map['label'])
  return '%s/%d/%s' % (MARKET_GAIN_DIR, config_map['predict_window'], market)

# Dir of training data/label/meta files.
def getDataDir(experiment_dir):
  return '%s/data' % experiment_dir

# Path to all data.
def getDataPath(data_dir):
  return '%s/data' % data_dir

# Path to all labels.
def getLabelPath(data_dir):
  return '%s/label' % data_dir

# Path to all regression labels.
def getRlabelPath(data_dir):
  return '%s/rlabel' % data_dir

# Path to all metadata.
def getMetaPath(data_dir):
  return '%s/meta' % data_dir

# Path to training metadata.
def getTrainingMetaPath(data_dir):
  return '%s/train_meta' % data_dir

# Path to prediction metadata.
def getPredictionMetaPath(data_dir):
  return '%s/predict_meta' % data_dir

# Path to training weights.
def getWeightPath(data_dir):
  return '%s/weight' % data_dir

# Dir of trained models.
def getModelDir(experiment_dir):
  return '%s/models' % experiment_dir

# Dir of imputers.
def getImputerDir(experiment_dir):
  return '%s/imputers' % experiment_dir

# Get model base name.
def getModelName(config_map):
  pos = config_map['model_spec'].find('(')
  assert pos > 0
  return config_map['model_spec'][:pos]

# Path to a specific model.
def getModelPath(model_dir, date, config_map):
  model = getModelName(config_map)
  return '%s/%s-%s-%d' % (model_dir, model, date, config_map['train_window'])

# Path to a specific imputer.
def getImputerPath(imputer_dir, date, config_map):
  return '%s/imputer-%s-%d' % (imputer_dir, date, config_map['train_window'])

# Dir of prediction results.
def getResultDir(expeirment_dir):
  return '%s/results' % expeirment_dir

# Path to the result file -- since we only support one version of models
# the name is simply 'result'.
def getResultPath(result_dir):
  return '%s/result' % result_dir

# Dir of analysis.
def getAnalyzeDir(experiment_dir):
  return '%s/analyze' % experiment_dir

# Path to training stats.
def getStatsPath(experiment_dir, config_map):
  model = getModelName(config_map)
  return '%s/%s-stats.tsv' % (experiment_dir, model)

def makeFeatureList(experiment_dir, config_map):
  groups = [item.strip() for item in config_map['features'].split('+')]
  features = set()
  for group in groups:
    with open('%s/%s' % (FEATURE_LIST_DIR, group), 'r') as fp:
      features.update([line for line in fp.read().splitlines()
                       if not line.startswith('#')])
  with open(getFeatureListPath(experiment_dir), 'w') as fp:
    for feature in sorted(features):
      print >> fp, feature

def collectData(experiment_dir, config_map):
  data_dir = getDataDir(experiment_dir)
  util.maybeMakeDir(data_dir)

  gain_dir = getLabelDir(config_map['label'])
  feature_list = getFeatureListPath(experiment_dir)
  data_file = getDataPath(data_dir)
  meta_file = getMetaPath(data_dir)

  cmd = ('%s/collect_data.py --gain_dir=%s --feature_base_dir=%s '
         '--feature_list=%s --feature_stats=%s --min_date=%s --max_date=%s '
         '--window=%d --min_feature_perc=%f --data_file=%s --meta_file=%s' % (
            CODE_DIR, gain_dir, FEATURE_DIR, feature_list, FEATURE_STATS_FILE,
            config_map['min_date'], config_map['max_date'],
            config_map['feature_window'], config_map['min_feature_perc'],
            data_file, meta_file))
  util.run(cmd)

def collectLabels(experiment_dir, config_map):
  data_dir = getDataDir(experiment_dir)
  meta_file = getMetaPath(data_dir)
  label_file = getLabelPath(data_dir)
  weight_file = getWeightPath(data_dir)
  if config_map['use_perc']:
    thresh_args = '--max_neg_perc=%f --min_pos_perc=%f' % (
        config_map['max_neg'], config_map['min_pos'])
  else:
    thresh_args = '--max_neg=%f --min_pos=%f' % (
        config_map['max_neg'], config_map['min_pos'])
  cmd = ('%s/collect_labels.py --meta_file=%s %s --label_file=%s '
         '--weight_power=%f --weight_file=%s' % (
            CODE_DIR, meta_file, thresh_args, label_file,
            config_map['weight_power'], weight_file))
  util.run(cmd)

def filterMetadata(experiment_dir, config, filter_str, label_file, filtered_path):
  filters = [filter.strip() for filter in filter_str.split('+')]
  filter_args = []
  _, market = getSourceAndMarket(config['label'])
  for filter in filters:
    if filter == '':
      continue
    key, value = filter.split('=')
    if key == 'min_raw_price':
      filter_args.append('--min_raw_price=%s' % value)
      if market == 'eod':
        filter_args.append('--raw_price_dir=%s' % EOD_PRICE_DIR)
      else:
        filter_args.append('--raw_price_dir=%s' % YAHOO_PRICE_DIR)
      continue
    if key == 'max_volatility_perc':
      filter_args.append('--max_volatility=%s' % value)
      if market == 'eod':
        filter_args.append('--volatility_dir=%s_%d' % (
            EOD_VOLATILITY_PERC_PREFIX, FILTER_VOLATILITY_K))
      else:
        filter_args.append('--volatility_dir=%s_%d' % (
            YAHOO_VOLATILITY_PERC_PREFIX, FILTER_VOLATILITY_K))
      continue
    if key == 'min_marketcap':
      filter_args.append('--min_marketcap=%s' % value)
      filter_args.append('--marketcap_dir=%s/MARKETCAP-ND' % FEATURE_DIR)
      continue
    if key == 'max_holes':
      filter_args.append('--max_holes=%s' % value)
      filter_args.append('--hole_dir=%s' % YAHOO_HOLE_DIR)
      continue
    if key == 'membership':
      assert value == MEMBERSHIP
      filter_args.append('--membership_file=%s' % MEMBERSHIP_FILE)
      continue
    assert False, 'unrecognized filter arg: %s' % filter

  data_dir = getDataDir(experiment_dir)
  meta_file = getMetaPath(data_dir)
  label_args = ''
  if label_file is not None:
    label_args = '--remove_neg_labels --label_file=%s' % label_file

  cmd = ('%s/filter_metadata.py --input_file=%s --output_file=%s %s %s' % (
      CODE_DIR, meta_file, filtered_path, ' '.join(filter_args), label_args))
  util.run(cmd)

def evaluateModel(model_file, imputer_file, data_file, label_file):
  with open(model_file, 'rb') as fp:
    model = pickle.load(fp)
  with open(imputer_file, 'rb') as fp:
    imputer = pickle.load(fp)
  X = numpy.loadtxt(data_file)
  y = numpy.loadtxt(label_file)

  X = imputer.transform(X)

  p = model.predict(X)

  result = dict()
  result['f1'] = f1_score(y, p)
  result['auc'] = roc_auc_score(y, p)

  yp = [[y[i], p[i]] for i in range(X.shape[0])]
  yp.sort(key=lambda item: item[1], reverse=True)

  recall_denom = sum(y)
  for perc in EVAL_PERCS:
    end = int(X.shape[0] * perc / 100)
    assert end > 0
    num = 0.0
    for i in range(end):
      if yp[i][0] > 0.5:
        num += 1
    precision = num / end
    recall = num / recall_denom
    result['%dperc-precision' % perc] = precision
    result['%dperc-recall' % perc] = recall

  return result

def trainModels(experiment_dir, config_map, train_meta_file):
  dates = []
  date = config_map['start_date']
  while date <= config_map['end_date']:
    dates.append(date)
    year = int(date[:4])
    month = int(date[4:])
    if month < 12:
      month += 1
    else:
      month = 1
      year += 1
    date = '%04d%02d' % (year, month)

  data_dir = getDataDir(experiment_dir)
  data_file = getDataPath(data_dir)
  if config_map['use_classification']:
    label_file = getLabelPath(data_dir)
  else:
    label_file = getRlabelPath(data_dir)
  meta_file = getMetaPath(data_dir)

  model_dir = getModelDir(experiment_dir)
  util.maybeMakeDir(model_dir)
  imputer_dir = getImputerDir(experiment_dir)
  util.maybeMakeDir(imputer_dir)

  stats_file = getStatsPath(experiment_dir, config_map)
  weight_args = ''
  if config_map['use_weight']:
    weight_args = '--weight_file=%s --tmp_weight_file=%s' % (
        getWeightPath(data_dir), TMP_WEIGHT_FILE)
  with open(stats_file, 'w') as fp:
    # Keep in sync with evaluateModel().
    print >> fp, '\t'.join([
        'date',
        'f1',
        'auc',
        '1perc-precision',
        '1perc-recall',
        '10perc-precision',
        '10perc-recall',
        '100perc-precision',
        '100perc-recall',
    ])
    for date in dates:
      model_file = getModelPath(model_dir, date, config_map)
      imputer_file = getImputerPath(imputer_dir, date, config_map)
      cmd = ('%s/train_model.py --data_file=%s --label_file=%s --meta_file=%s %s '
             '--yyyymm=%s --months=%d --model_def="%s" --perc=%f --model_file=%s '
             '--train_meta_file=%s --tmp_data_file=%s --tmp_label_file=%s '
             '--imputer_strategy=%s --imputer_file=%s' % (
                CODE_DIR, data_file, label_file, meta_file, weight_args, date,
                config_map['train_window'], config_map['model_spec'],
                config_map['train_perc'], model_file, train_meta_file,
                TMP_DATA_FILE, TMP_LABEL_FILE,
                config_map['imputer_strategy'], imputer_file))
      util.run(cmd)
      if not os.path.isfile(model_file):
        continue
      if config_map['use_classification']:
        result = evaluateModel(model_file, imputer_file, TMP_DATA_FILE, TMP_LABEL_FILE)
        # Keep in sync with evaluateModel().
        values = [date, '%.4f' % result['f1'], '%.4f' % result['auc']]
        for perc in EVAL_PERCS:
          values.append('%.4f' % result['%dperc-precision' % perc])
          values.append('%.4f' % result['%dperc-recall' % perc])
        print >> fp, '\t'.join(values)
        fp.flush()

def predict(experiment_dir, config_map, predict_meta_file):
  result_dir = getResultDir(experiment_dir)
  util.maybeMakeDir(result_dir)
  result_file = getResultPath(result_dir)

  data_dir = getDataDir(experiment_dir)
  data_file = getDataPath(data_dir)
  if config_map['use_classification']:
    label_file = getLabelPath(data_dir)
  else:
    label_file = getRlabelPath(data_dir)
  meta_file = getMetaPath(data_dir)
  model_dir = getModelDir(experiment_dir)
  imputer_dir = getImputerDir(experiment_dir)

  model_prefix = '%s-' % getModelName(config_map)
  model_suffix = '-%d' % config_map['train_window']
  imputer_prefix = 'imputer-'
  imputer_suffix = '-%d' % config_map['train_window']

  cmd = ('%s/predict_all.py --data_file=%s --label_file=%s '
         '--meta_file=%s --model_dir=%s --model_prefix="%s" '
         '--model_suffix="%s" --imputer_dir=%s --imputer_prefix="%s" '
         '--imputer_suffix="%s" --prediction_window=%d '
         '--delay_window=%d --predict_meta_file=%s --result_file=%s' % (
            CODE_DIR, data_file, label_file, meta_file,
            model_dir, model_prefix, model_suffix,
            imputer_dir, imputer_prefix, imputer_suffix,
            config_map['predict_window'],
            config_map['delay_window'], predict_meta_file,
            result_file))
  util.run(cmd)

def analyze(experiment_dir, config_map):
  market_gain_file = getMarketGainPath(config_map)
  analyze_dir = getAnalyzeDir(experiment_dir)
  util.maybeMakeDir(analyze_dir)
  result_dir = getResultDir(experiment_dir)
  result_file = getResultPath(result_dir)
  cmd = ('%s/analyze_all.py --result_file=%s --hold_period=%d '
         '--analyze_dir=%s --market_gain_file=%s' % (
            CODE_DIR, result_file, config_map['predict_window'],
            analyze_dir, market_gain_file))
  util.run(cmd)

def runExperiment(config_file):
  assert config_file.endswith(CONFIG_SUFFIX)
  pos = config_file.rfind('/')
  assert pos > 0
  experiment = config_file[pos+1:-len(CONFIG_SUFFIX)]
  experiment_dir = '%s/%s' % (EXPERIMENT_BASE_DIR, experiment)
  util.maybeMakeDir(experiment_dir)
  config_map = getConfig(config_file)

  makeFeatureList(experiment_dir, config_map)

  step = '%s_collect_data' % experiment
  if not util.checkDone(step):
    collectData(experiment_dir, config_map)
    util.markDone(step)

  step = '%s_collect_label' % experiment
  if not util.checkDone(step):
    collectLabels(experiment_dir, config_map)
    util.markDone(step)

  data_dir = getDataDir(experiment_dir)
  if config_map['use_classification']:
    # For classification, negative labels indicate gain between
    # max_neg and min_pos and should be removed from training
    # (but not prediction).
    label_file = getLabelPath(data_dir)
  else:
    # Do not filter out negative regression labels.
    label_file = None
  train_meta_file = getTrainingMetaPath(data_dir)
  predict_meta_file = getPredictionMetaPath(data_dir)

  step = '%s_filter_train' % experiment
  if not util.checkDone(step):
    filterMetadata(experiment_dir, config_map,
                   config_map['train_filter'], label_file, train_meta_file)
    util.markDone(step)

  step = '%s_filter_predict' % experiment
  if not util.checkDone(step):
    filterMetadata(experiment_dir, config_map,
                   config_map['predict_filter'], None, predict_meta_file)
    util.markDone(step)

  step = '%s_train_models' % experiment
  if not util.checkDone(step):
    trainModels(experiment_dir, config_map, train_meta_file)
    util.markDone(step)

  step = '%s_predict' % experiment
  if not util.checkDone(step):
    predict(experiment_dir, config_map, predict_meta_file)
    util.markDone(step)

  step = '%s_analyze' % experiment
  if not util.checkDone(step):
    analyze(experiment_dir, config_map)
    util.markDone(step)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', required=True)
  args = parser.parse_args()
  util.configLogging()
  runExperiment(args.config)

if __name__ == '__main__':
  main()

