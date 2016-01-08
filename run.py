#!/usr/bin/python2.7

from config import *
import logging
import os
import util

LOG_LEVEL = logging.INFO

# Steps to execute on local machine, set to False to skip steps.
DO_LOCAL = {
    'get_sf1_tickers': True,
    'download_yahoo': True,
}

# Steps to execute on remote machine, set to False to skip steps.
DO_EOD = False
DO_REMOTE = {
    'get_sf1_tickers': True,
    'get_eod_tickers': DO_EOD,
    'convert_sf1_raw': True,
    'process_sf1_raw': True,
    'convert_eod_raw': DO_EOD,
    'process_eod_raw': DO_EOD,
    'process_yahoo': True,

    # Need access to raw-format files, thus not compatible to EOD.
    'get_yahoo_trading_days': True,
    'get_yahoo_holes': True,

    'compute_basic_features': True,
    'compute_basic_features_mrx': True,
    'compute_custom_features': True,
    'compute_custom_features_mrx': True,

    # Disabled vert perc and variants of hari perc features,
    # except for compute_hori_perc_features.
    'compute_vert_perc_features': False,
    'compute_vert_perc_2_features': False,
    'compute_hori_perc_features': True,
    'compute_hori_rank_perc_features': False,
    'compute_hori_perc_features_sector': False,
    'compute_hori_rank_perc_features_sector': False,

    'compute_hori_perc_features_mrx': True,
    'compute_vert_gain_features': False,

    'get_feature_stats': True,

    # Disabled sector/industry features.
    'get_sector_map': False,
    'get_industry_map': False,
    'get_sector_feature': False,
    'get_industry_feature': False,

    'get_eod_price': DO_EOD,
    'get_eod_adjprice': DO_EOD,
    'get_eod_logprice': DO_EOD,
    'get_eod_logadjprice': DO_EOD,
    'get_eod_logadjvolume': DO_EOD,
    'get_yahoo_price': True,
    'get_yahoo_adjprice': True,

    # Disabled log price/volume features.
    'get_yahoo_logprice': False,
    'get_yahoo_logadjprice': False,
    'get_yahoo_logadjvolume': False,
    'get_eod_gain_feature': DO_EOD,

    # Disabled gain features.
    'get_yahoo_gain_feature': False,

    # Disabled membership filtering.
    'get_membership': False,  # Files changed on 2015-10-01. Need to be fixed.

    'get_eod_gain_label': DO_EOD,
    'get_yahoo_gain_label': True,
    # Only yahoo offers market index history, so there is no eod version.
    'process_market': True,
    'get_market_adjprice': True,
    'get_market_gain': True,

    # Diabled egain features.
    'get_eod_egain_feature': DO_EOD,
    'get_yahoo_egain_feature': False,

    'get_eod_egain_label': DO_EOD,
    'get_yahoo_egain_label': True,

    # Diabled price/volume features.
    'compute_eod_logprice_feature': DO_EOD,
    'compute_yahoo_logprice_feature': False,
    'compute_eod_logadjprice_feature': DO_EOD,
    'compute_yahoo_logadjprice_feature': False,
    'compute_eod_logadjvolume_feature': DO_EOD,
    'compute_yahoo_logadjvolume_feature': False,
    'compute_eod_adjprice_feature': DO_EOD,
    'compute_yahoo_adjprice_feature': False,
    'compute_eod_price_feature': DO_EOD,
    'compute_yahoo_price_feature': False,
    'compute_eod_gain_feature': DO_EOD,
    'compute_yahoo_gain_feature': False,
    'compute_eod_egain_feature': DO_EOD,
    'compute_yahoo_egain_feature': False,

    'compute_eod_volatility': DO_EOD,
    'compute_yahoo_volatility': True,
    'compute_eod_volatility_perc': DO_EOD,
    'compute_yahoo_volatility_perc': True,
}

if TEST:
  DO = DO_REMOTE
elif HOST == 'lnyang-mn1':
  DO = DO_LOCAL
else:
  DO = DO_REMOTE

####################
## Util functions ##
####################

# Checks to run or skip specified step, logs and returns decision.
def logDo(step):
  if DO.get(step, False) and not util.checkDone(step):
    logging.info('running step: %s' % step)
    return True
  logging.info('skipping step: %s' % step)
  return False

# Shortcut to util.run() with dry run option.
def run(cmd, step=None):
  util.run(cmd, dry_run=DRY_RUN, step=step)

# Shortcut to util.markDone().
def markDone(step):
  util.markDone(step)

############
## Script ##
############

util.configLogging(LOG_LEVEL)

# Prepare dirs.
util.maybeMakeDirs([
    SYMBOL_DIR,
    TICKER_DIR,
    YAHOO_SF1_DIR,
    SF1_RAW_DIR,
    SF1_PROCESSED_DIR,
    EOD_RAW_DIR,
    EOD_PROCESSED_DIR,
    YAHOO_PROCESSED_DIR,
    FEATURE_DIR,
    FEATURE_INFO_DIR,
    MISC_DIR,
    EOD_PRICE_DIR,
    EOD_ADJPRICE_DIR,
    EOD_LOGPRICE_DIR,
    EOD_LOGADJPRICE_DIR,
    EOD_LOGADJVOLUME_DIR,
    YAHOO_PRICE_DIR,
    YAHOO_ADJPRICE_DIR,
    YAHOO_LOGPRICE_DIR,
    YAHOO_LOGADJPRICE_DIR,
    YAHOO_LOGADJVOLUME_DIR,
    YAHOO_HOLE_DIR,
    EOD_GAIN_DIR,
    YAHOO_GAIN_DIR,
    EOD_GAIN_LABEL_DIR,
    YAHOO_GAIN_LABEL_DIR,
    MARKET_PROCESSED_DIR,
    MARKET_ADJPRICE_DIR,
    MARKET_GAIN_DIR,
    EOD_EGAIN_DIR,
    YAHOO_EGAIN_DIR,
    EOD_EGAIN_LABEL_DIR,
    YAHOO_EGAIN_LABEL_DIR,
])

if logDo('get_sf1_tickers'):
  cmd = '%s/get_sf1_tickers.py --sf1_file=%s --info_file=%s --ticker_file=%s' % (
      CODE_DIR, RAW_SF1_FILE, SF1_INFO_FILE, SF1_TICKER_FILE)
  run(cmd, 'get_sf1_tickers')

if logDo('get_eod_tickers'):
  cmd = '%s/get_eod_tickers.py --eod_file=%s --ticker_file=%s' % (
      CODE_DIR, RAW_EOD_FILE, EOD_TICKER_FILE)
  run(cmd, 'get_eod_tickers')

if logDo('download_yahoo'):
  cmd = '%s/download_yahoo.py --ticker_file=%s --download_dir=%s' % (
      CODE_DIR, SF1_TICKER_FILE, YAHOO_SF1_DIR)
  run(cmd, 'download_yahoo')

if logDo('convert_sf1_raw'):
  cmd = ('%s/convert_sf1_raw.py --sf1_file=%s --indicator_file=%s '
         '--raw_dir=%s' % (
             CODE_DIR, RAW_SF1_FILE, SF1_INDICATOR_FILE, SF1_RAW_DIR))
  run(cmd, 'convert_sf1_raw')

if logDo('process_sf1_raw'):
  cmd = '%s/process_sf1_raw.py --raw_dir=%s --processed_dir=%s' % (
      CODE_DIR, SF1_RAW_DIR, SF1_PROCESSED_DIR)
  run(cmd, 'process_sf1_raw')

if logDo('convert_eod_raw'):
  cmd = '%s/convert_eod_raw.py --eod_file=%s --raw_dir=%s' % (
      CODE_DIR, RAW_EOD_FILE, EOD_RAW_DIR)
  run(cmd, 'convert_eod_raw')

if logDo('process_eod_raw'):
  cmd = ('%s/process_eod_raw.py --raw_dir=%s --ticker_file=%s '
         '--processed_dir=%s' % (
             CODE_DIR, EOD_RAW_DIR, SF1_TICKER_FILE, EOD_PROCESSED_DIR))
  run(cmd, 'process_eod_raw')

if logDo('process_yahoo'):
  cmd = '%s/process_yahoo.py --raw_dir=%s --processed_dir=%s' % (
      CODE_DIR, YAHOO_SF1_DIR, YAHOO_PROCESSED_DIR)
  run(cmd, 'process_yahoo')

if logDo('get_yahoo_trading_days'):
  cmd = '%s/get_yahoo_trading_days.py --raw_dir=%s --output_file=%s' % (
      CODE_DIR, YAHOO_SF1_DIR, YAHOO_TRADING_DAY_FILE)
  run(cmd, 'get_yahoo_trading_days')

if logDo('get_yahoo_holes'):
  cmd = ('%s/get_yahoo_holes.py --raw_dir=%s --trading_day_file=%s '
         '--output_dir=%s' % (
      CODE_DIR, YAHOO_SF1_DIR, YAHOO_TRADING_DAY_FILE, YAHOO_HOLE_DIR))
  run(cmd, 'get_yahoo_holes')

if logDo('compute_basic_features'):
  cmd = ('%s/compute_basic_features.py --processed_dir=%s --ticker_file=%s '
         '--feature_base_dir=%s --info_dir=%s '
         '--computer=%s/compute_basic_feature.py') % (
      CODE_DIR, SF1_PROCESSED_DIR, SF1_TICKER_FILE,
      FEATURE_DIR, FEATURE_INFO_DIR, CODE_DIR)
  run(cmd, 'compute_basic_features')

if logDo('compute_basic_features_mrx'):
  cmd = ('%s/compute_basic_features.py --processed_dir=%s --ticker_file=%s '
         '--feature_base_dir=%s --info_dir=%s '
         '--computer=%s/compute_basic_feature.py --use_mrx') % (
      CODE_DIR, SF1_PROCESSED_DIR, SF1_TICKER_FILE,
      FEATURE_DIR, FEATURE_INFO_DIR, CODE_DIR)
  run(cmd, 'compute_basic_features_mrx')

if logDo('compute_custom_features'):
  cmd = ('%s/compute_custom_features.py --feature_base_dir=%s --ticker_file=%s '
         '--info_dir=%s --computer=%s/compute_custom_feature.py') % (
      CODE_DIR, FEATURE_DIR, SF1_TICKER_FILE, FEATURE_INFO_DIR, CODE_DIR)
  run(cmd, 'compute_custom_features')

if logDo('compute_custom_features_mrx'):
  cmd = ('%s/compute_custom_features.py --feature_base_dir=%s --ticker_file=%s '
         '--info_dir=%s --computer=%s/compute_custom_feature.py --use_mrx') % (
      CODE_DIR, FEATURE_DIR, SF1_TICKER_FILE, FEATURE_INFO_DIR, CODE_DIR)
  run(cmd, 'compute_custom_features_mrx')

if logDo('compute_vert_perc_features'):
  cmd = ('%s/compute_vert_perc_features.py --sf1_input_dir=%s --price_input_dir=%s '
         '--feature_base_dir=%s --ticker_file=%s --info_dir=%s '
         '--computer=%s/compute_vert_perc_feature.py') % (
      CODE_DIR, FEATURE_DIR, YAHOO_DIR, FEATURE_DIR, SF1_TICKER_FILE,
      FEATURE_INFO_DIR, CODE_DIR)
  run(cmd, 'compute_vert_perc_features')

if logDo('compute_vert_perc_2_features'):
  cmd = ('%s/compute_vert_perc_2_features.py --sf1_input_dir=%s --price_input_dir=%s '
         '--feature_base_dir=%s --ticker_file=%s --info_dir=%s '
         '--computer=%s/compute_vert_perc_2_feature.py') % (
      CODE_DIR, FEATURE_DIR, YAHOO_DIR, FEATURE_DIR, SF1_TICKER_FILE,
      FEATURE_INFO_DIR, CODE_DIR)
  run(cmd, 'compute_vert_perc_2_features')

if logDo('compute_hori_perc_features'):
  cmd = ('%s/compute_hori_perc_features.py --feature_base_dir=%s '
         '--suffix=_hp --computer=%s/compute_hori_perc_feature.py') % (
      CODE_DIR, FEATURE_DIR, CODE_DIR)
  run(cmd, 'compute_hori_perc_features')

if logDo('compute_hori_perc_features_mrx'):
  cmd = ('%s/compute_hori_perc_features.py --feature_base_dir=%s '
         '--suffix=_hp --computer=%s/compute_hori_perc_feature.py --use_mrx') % (
      CODE_DIR, FEATURE_DIR, CODE_DIR)
  run(cmd, 'compute_hori_perc_features_mrx')

if logDo('compute_hori_rank_perc_features'):
  cmd = ('%s/compute_hori_perc_features.py --feature_base_dir=%s --rank '
         '--suffix=_hpr --computer=%s/compute_hori_perc_feature.py') % (
      CODE_DIR, FEATURE_DIR, CODE_DIR)
  run(cmd, 'compute_hori_rank_perc_features')

if logDo('get_sector_map'):
  cmd = ('%s/get_sector_industry_map.py --ticker_file=%s --info_file=%s '
         '--sector --map_file=%s --stats_file=%s' % (
      CODE_DIR, SF1_TICKER_FILE, SF1_INFO_FILE,
      SECTOR_MAP_FILE, SECTOR_STATS_FILE))
  run(cmd, 'get_sector_map')

if logDo('get_industry_map'):
  cmd = ('%s/get_sector_industry_map.py --ticker_file=%s --info_file=%s '
         '--industry --map_file=%s --stats_file=%s' % (
      CODE_DIR, SF1_TICKER_FILE, SF1_INFO_FILE,
      INDUSTRY_MAP_FILE, INDUSTRY_STATS_FILE))
  run(cmd, 'get_industry_map')

if logDo('compute_hori_perc_features_sector'):
  cmd = ('%s/compute_hori_perc_features.py --feature_base_dir=%s --suffix=_hp_sector '
         '--group_map_file=%s --computer=%s/compute_hori_perc_feature.py') % (
      CODE_DIR, FEATURE_DIR, SECTOR_MAP_FILE, CODE_DIR)
  run(cmd, 'compute_hori_perc_features_sector')

if logDo('compute_hori_rank_perc_features_sector'):
  cmd = ('%s/compute_hori_perc_features.py --feature_base_dir=%s '
         '--group_map_file=%s --rank --suffix=_hpr_sector '
         '--computer=%s/compute_hori_perc_feature.py') % (
      CODE_DIR, FEATURE_DIR, SECTOR_MAP_FILE, CODE_DIR)
  run(cmd, 'compute_hori_rank_perc_features_sector')

if logDo('compute_vert_gain_features'):
  cmd = ('%s/compute_vert_gain_features.py --sf1_input_dir=%s '
         '--feature_base_dir=%s --ticker_file=%s --info_dir=%s '
         '--computer=%s/compute_vert_gain_feature.py') % (
      CODE_DIR, FEATURE_DIR, FEATURE_DIR, SF1_TICKER_FILE,
      FEATURE_INFO_DIR, CODE_DIR)
  run(cmd, 'compute_vert_gain_features')

if logDo('get_feature_stats'):
  cmd = '%s/get_feature_stats.py --info_dir=%s --stats_file=%s' % (
      CODE_DIR, FEATURE_INFO_DIR, FEATURE_STATS_FILE)
  run(cmd, 'get_feature_stats')

if logDo('get_sector_feature'):
  cmd = ('%s/get_sector_industry_feature.py --map_file=%s --sector '
         '--output_base_dir=%s' % (
      CODE_DIR, SECTOR_MAP_FILE, FEATURE_DIR))
  run(cmd, 'get_sector_feature')

if logDo('get_industry_feature'):
  cmd = ('%s/get_sector_industry_feature.py --map_file=%s --industry '
         '--output_base_dir=%s' % (
      CODE_DIR, INDUSTRY_MAP_FILE, FEATURE_DIR))
  run(cmd, 'get_industry_feature')

if logDo('get_eod_price'):
  cmd = ('%s/get_price_volume.py --processed_dir=%s --column=price '
         '--output_dir=%s' % (
      CODE_DIR, EOD_PROCESSED_DIR, EOD_PRICE_DIR))
  run(cmd, 'get_eod_price')

if logDo('get_eod_adjprice'):
  cmd = ('%s/get_price_volume.py --processed_dir=%s --column=adjprice '
         '--output_dir=%s' % (
      CODE_DIR, EOD_PROCESSED_DIR, EOD_ADJPRICE_DIR))
  run(cmd, 'get_eod_adjprice')

if logDo('get_eod_logprice'):
  cmd = ('%s/get_price_volume.py --processed_dir=%s --column=price '
         '--take_log --output_dir=%s' % (
      CODE_DIR, EOD_PROCESSED_DIR, EOD_LOGPRICE_DIR))
  run(cmd, 'get_eod_logprice')

if logDo('get_eod_logadjprice'):
  cmd = ('%s/get_price_volume.py --processed_dir=%s --column=adjprice '
         '--take_log --output_dir=%s' % (
      CODE_DIR, EOD_PROCESSED_DIR, EOD_LOGADJPRICE_DIR))
  run(cmd, 'get_eod_logadjprice')

if logDo('get_eod_logadjvolume'):
  cmd = ('%s/get_price_volume.py --processed_dir=%s --column=adjvolume '
         '--take_log --output_dir=%s' % (
      CODE_DIR, EOD_PROCESSED_DIR, EOD_LOGADJVOLUME_DIR))
  run(cmd, 'get_eod_logadjvolume')

if logDo('get_yahoo_price'):
  cmd = ('%s/get_price_volume.py --processed_dir=%s --column=price '
         '--output_dir=%s' % (
      CODE_DIR, YAHOO_PROCESSED_DIR, YAHOO_PRICE_DIR))
  run(cmd, 'get_yahoo_price')

if logDo('get_yahoo_adjprice'):
  cmd = ('%s/get_price_volume.py --processed_dir=%s --column=adjprice '
         '--output_dir=%s' % (
      CODE_DIR, YAHOO_PROCESSED_DIR, YAHOO_ADJPRICE_DIR))
  run(cmd, 'get_yahoo_adjprice')

if logDo('get_yahoo_logprice'):
  cmd = ('%s/get_price_volume.py --processed_dir=%s --column=price '
         '--take_log --output_dir=%s' % (
      CODE_DIR, YAHOO_PROCESSED_DIR, YAHOO_LOGPRICE_DIR))
  run(cmd, 'get_yahoo_logprice')

if logDo('get_yahoo_logadjprice'):
  cmd = ('%s/get_price_volume.py --processed_dir=%s --column=adjprice '
         '--take_log --output_dir=%s' % (
      CODE_DIR, YAHOO_PROCESSED_DIR, YAHOO_LOGADJPRICE_DIR))
  run(cmd, 'get_yahoo_logadjprice')

if logDo('get_yahoo_logadjvolume'):
  cmd = ('%s/get_price_volume.py --processed_dir=%s --column=adjvolume '
         '--take_log --output_dir=%s' % (
      CODE_DIR, YAHOO_PROCESSED_DIR, YAHOO_LOGADJVOLUME_DIR))
  run(cmd, 'get_yahoo_logadjvolume')

if logDo('get_eod_gain_feature'):
  for k in GAIN_K_LIST:
    gain_dir = '%s/%d' % (EOD_GAIN_DIR, k)
    util.maybeMakeDir(gain_dir)
    cmd = '%s/compute_gain.py --price_dir=%s --k=%d --gain_dir=%s' % (
        CODE_DIR, EOD_ADJPRICE_DIR, k, gain_dir)
    run(cmd)
  markDone('get_eod_gain_feature')

if logDo('get_yahoo_gain_feature'):
  for k in GAIN_K_LIST:
    gain_dir = '%s/%d' % (YAHOO_GAIN_DIR, k)
    util.maybeMakeDir(gain_dir)
    cmd = '%s/compute_gain.py --price_dir=%s --k=%d --gain_dir=%s' % (
        CODE_DIR, YAHOO_ADJPRICE_DIR, k, gain_dir) 
    run(cmd)
  markDone('get_yahoo_gain_feature')

if logDo('get_membership'):
  cmd = ('%s/get_membership.py --boy_file=%s/%s-boy.tsv '
         '--change_file=%s/%s-changes.tsv --membership_file=%s' % (
      CODE_DIR, RAW_MISC_DIR, MEMBERSHIP, RAW_MISC_DIR, MEMBERSHIP,
      MEMBERSHIP_FILE))
  run(cmd)
  markDone('get_membership')

if logDo('get_eod_gain_label'):
  cmd = '%s/compute_gain.py --price_dir=%s --k=%d --fill --gain_dir=%s' % (
      CODE_DIR, EOD_ADJPRICE_DIR, PREDICTION_WINDOW, EOD_GAIN_LABEL_DIR)
  run(cmd, 'get_eod_gain_label')

if logDo('get_yahoo_gain_label'):
  cmd = '%s/compute_gain.py --price_dir=%s --k=%d --fill --gain_dir=%s' % (
      CODE_DIR, YAHOO_ADJPRICE_DIR, PREDICTION_WINDOW, YAHOO_GAIN_LABEL_DIR)
  run(cmd, 'get_yahoo_gain_label')

if logDo('process_market'):
  cmd = '%s/process_yahoo.py --raw_dir=%s --processed_dir=%s' % (
      CODE_DIR, YAHOO_MARKET_DIR, MARKET_PROCESSED_DIR)
  run(cmd, 'process_market')

if logDo('get_market_adjprice'):
  cmd = ('%s/get_price_volume.py --processed_dir=%s --column=adjprice '
         '--output_dir=%s' % (
      CODE_DIR, MARKET_PROCESSED_DIR, MARKET_ADJPRICE_DIR))
  run(cmd, 'get_market_adjprice')

if logDo('get_market_gain'):
  # For market we only do one version of gain (without mininum raw price)
  # and it will be used for both features and labels.  GAIN_K_LIST specify
  # all the windows for features, and [PREDICTION_WINDOW] specify those
  # for labels.  (Since PREDICTION_WINDOW = 12 which is in GAIN_K_LIST
  # this is no-op for now.)
  k_list = set(GAIN_K_LIST + [PREDICTION_WINDOW])
  for k in sorted(k_list):
    gain_dir = '%s/%d' % (MARKET_GAIN_DIR, k)
    util.maybeMakeDir(gain_dir)
    cmd = '%s/compute_gain.py --price_dir=%s --k=%d --fill --gain_dir=%s' % (
        CODE_DIR, MARKET_ADJPRICE_DIR, k, gain_dir)
    run(cmd)
  markDone('get_market_gain')

if logDo('get_eod_egain_feature'):
  for k in GAIN_K_LIST:
    gain_dir = '%s/%d' % (EOD_GAIN_DIR, k)
    for market in MARKETS:
      market_file = '%s/%d/%s' % (MARKET_GAIN_DIR, k, market)
      egain_dir = '%s/%d/%s' % (EOD_EGAIN_DIR, k, market)
      util.maybeMakeDir(egain_dir)
      cmd = ('%s/compute_egain.py --gain_dir=%s --market_file=%s '
             '--egain_dir=%s' % (CODE_DIR, gain_dir, market_file, egain_dir))
      run(cmd)
  markDone('get_eod_egain_feature')

if logDo('get_yahoo_egain_feature'):
  for k in GAIN_K_LIST:
    gain_dir = '%s/%d' % (YAHOO_GAIN_DIR, k)
    for market in MARKETS:
      market_file = '%s/%d/%s' % (MARKET_GAIN_DIR, k, market)
      egain_dir = '%s/%d/%s' % (YAHOO_EGAIN_DIR, k, market)
      util.maybeMakeDir(egain_dir)
      cmd = ('%s/compute_egain.py --gain_dir=%s --market_file=%s '
             '--egain_dir=%s' % (CODE_DIR, gain_dir, market_file, egain_dir))
      run(cmd)
  markDone('get_yahoo_egain_feature')

if logDo('get_eod_egain_label'):
  for market in MARKETS:
    market_file = '%s/%d/%s' % (MARKET_GAIN_DIR, PREDICTION_WINDOW, market)
    egain_dir = '%s/%s' % (EOD_EGAIN_LABEL_DIR, market)
    util.maybeMakeDir(egain_dir)
    cmd = ('%s/compute_egain.py --gain_dir=%s --market_file=%s '
           '--egain_dir=%s' % (
        CODE_DIR, EOD_GAIN_LABEL_DIR, market_file, egain_dir))
    run(cmd)
  markDone('get_eod_egain_label')

if logDo('get_yahoo_egain_label'):
  for market in MARKETS:
    market_file = '%s/%d/%s' % (MARKET_GAIN_DIR, PREDICTION_WINDOW, market)
    egain_dir = '%s/%s' % (YAHOO_EGAIN_LABEL_DIR, market)
    util.maybeMakeDir(egain_dir)
    cmd = ('%s/compute_egain.py --gain_dir=%s --market_file=%s '
           '--egain_dir=%s' % (
        CODE_DIR, YAHOO_GAIN_LABEL_DIR, market_file, egain_dir))
    run(cmd)
  markDone('get_yahoo_egain_label')

if logDo('compute_eod_logprice_feature'):
  for k in LOGPRICE_K_LIST:
    output_dir = '%s/eod-logprice-%d' % (FEATURE_DIR, k)
    util.maybeMakeDir(output_dir)
    cmd = ('%s/compute_previous_feature.py --feature_dir=%s --k=%d '
           '--pfeature_dir=%s' % (
        CODE_DIR, EOD_LOGPRICE_DIR, k, output_dir))
    run(cmd)
  markDone('compute_eod_logprice_feature')

if logDo('compute_yahoo_logprice_feature'):
  for k in LOGPRICE_K_LIST:
    output_dir = '%s/yahoo-logprice-%d' % (FEATURE_DIR, k)
    util.maybeMakeDir(output_dir)
    cmd = ('%s/compute_previous_feature.py --feature_dir=%s --k=%d '
           '--pfeature_dir=%s' % (
        CODE_DIR, YAHOO_LOGPRICE_DIR, k, output_dir))
    run(cmd)
  markDone('compute_yahoo_logprice_feature')

if logDo('compute_eod_logadjprice_feature'):
  for k in LOGADJPRICE_K_LIST:
    output_dir = '%s/eod-logadjprice-%d' % (FEATURE_DIR, k)
    util.maybeMakeDir(output_dir)
    cmd = ('%s/compute_previous_feature.py --feature_dir=%s --k=%d '
           '--pfeature_dir=%s' % (
        CODE_DIR, EOD_LOGADJPRICE_DIR, k, output_dir))
    run(cmd)
  markDone('compute_eod_logadjprice_feature')

if logDo('compute_yahoo_logadjprice_feature'):
  for k in LOGADJPRICE_K_LIST:
    output_dir = '%s/yahoo-logadjprice-%d' % (FEATURE_DIR, k)
    util.maybeMakeDir(output_dir)
    cmd = ('%s/compute_previous_feature.py --feature_dir=%s --k=%d '
           '--pfeature_dir=%s' % (
        CODE_DIR, YAHOO_LOGADJPRICE_DIR, k, output_dir))
    run(cmd)
  markDone('compute_yahoo_logadjprice_feature')

if logDo('compute_eod_logadjvolume_feature'):
  for k in LOGADJVOLUME_K_LIST:
    output_dir = '%s/eod-logadjvolume-%d' % (FEATURE_DIR, k)
    util.maybeMakeDir(output_dir)
    cmd = ('%s/compute_previous_feature.py --feature_dir=%s --k=%d '
           '--pfeature_dir=%s' % (
        CODE_DIR, EOD_LOGADJVOLUME_DIR, k, output_dir))
    run(cmd)
  markDone('compute_eod_logadjvolume_feature')

if logDo('compute_yahoo_logadjvolume_feature'):
  for k in LOGADJVOLUME_K_LIST:
    output_dir = '%s/yahoo-logadjvolume-%d' % (FEATURE_DIR, k)
    util.maybeMakeDir(output_dir)
    cmd = ('%s/compute_previous_feature.py --feature_dir=%s --k=%d '
           '--pfeature_dir=%s' % (
        CODE_DIR, YAHOO_LOGADJVOLUME_DIR, k, output_dir))
    run(cmd)
  markDone('compute_yahoo_logadjvolume_feature')

if logDo('compute_eod_adjprice_feature'):
  for k in ADJPRICE_K_LIST:
    output_dir = '%s/eod-adjprice-%d' % (FEATURE_DIR, k)
    util.maybeMakeDir(output_dir)
    cmd = ('%s/compute_previous_feature.py --feature_dir=%s --k=%d '
           '--pfeature_dir=%s' % (
        CODE_DIR, EOD_ADJPRICE_DIR, k, output_dir))
    run(cmd)
  markDone('compute_eod_adjprice_feature')

if logDo('compute_yahoo_adjprice_feature'):
  for k in ADJPRICE_K_LIST:
    output_dir = '%s/yahoo-adjprice-%d' % (FEATURE_DIR, k)
    util.maybeMakeDir(output_dir)
    cmd = ('%s/compute_previous_feature.py --feature_dir=%s --k=%d '
           '--pfeature_dir=%s' % (
        CODE_DIR, YAHOO_ADJPRICE_DIR, k, output_dir))
    run(cmd)
  markDone('compute_yahoo_adjprice_feature')

if logDo('compute_eod_price_feature'):
  for k in PRICE_K_LIST:
    output_dir = '%s/eod-price-%d' % (FEATURE_DIR, k)
    util.maybeMakeDir(output_dir)
    cmd = ('%s/compute_previous_feature.py --feature_dir=%s --k=%d '
           '--pfeature_dir=%s' % (
        CODE_DIR, EOD_PRICE_DIR, k, output_dir))
    run(cmd)
  markDone('compute_eod_price_feature')

if logDo('compute_yahoo_price_feature'):
  for k in PRICE_K_LIST:
    output_dir = '%s/yahoo-price-%d' % (FEATURE_DIR, k)
    util.maybeMakeDir(output_dir)
    cmd = ('%s/compute_previous_feature.py --feature_dir=%s --k=%d '
           '--pfeature_dir=%s' % (
        CODE_DIR, YAHOO_PRICE_DIR, k, output_dir))
    run(cmd)
  markDone('compute_yahoo_price_feature')

if logDo('compute_eod_gain_feature'):
  for k in GAIN_K_LIST:
    input_dir = '%s/%d' % (EOD_GAIN_DIR, k)
    output_dir = '%s/eod-gain-%d' % (FEATURE_DIR, k)
    util.maybeMakeDir(output_dir)
    cmd = ('%s/compute_previous_feature.py --feature_dir=%s --k=%d '
           '--pfeature_dir=%s' % (
        CODE_DIR, input_dir, k, output_dir))
    run(cmd)
  markDone('compute_eod_gain_feature')

if logDo('compute_yahoo_gain_feature'):
  for k in GAIN_K_LIST:
    input_dir = '%s/%d' % (YAHOO_GAIN_DIR, k)
    output_dir = '%s/yahoo-gain-%d' % (FEATURE_DIR, k)
    util.maybeMakeDir(output_dir)
    cmd = ('%s/compute_previous_feature.py --feature_dir=%s --k=%d '
           '--pfeature_dir=%s' % (
        CODE_DIR, input_dir, k, output_dir))
    run(cmd)
  markDone('compute_yahoo_gain_feature')

if logDo('compute_eod_egain_feature'):
  for k in GAIN_K_LIST:
    for market in MARKETS:
      input_dir = '%s/%d/%s' % (EOD_EGAIN_DIR, k, market)
      output_dir = '%s/eod-%s-egain-%d' % (FEATURE_DIR, market, k)
      util.maybeMakeDir(output_dir)
      cmd = ('%s/compute_previous_feature.py --feature_dir=%s --k=%d '
             '--pfeature_dir=%s' % (
          CODE_DIR, input_dir, k, output_dir))
      run(cmd)
  markDone('compute_eod_egain_feature')

if logDo('compute_yahoo_egain_feature'):
  for k in GAIN_K_LIST:
    for market in MARKETS:
      input_dir = '%s/%d/%s' % (YAHOO_EGAIN_DIR, k, market)
      output_dir = '%s/yahoo-%s-egain-%d' % (FEATURE_DIR, market, k)
      util.maybeMakeDir(output_dir)
      cmd = ('%s/compute_previous_feature.py --feature_dir=%s --k=%d '
             '--pfeature_dir=%s' % (
          CODE_DIR, input_dir, k, output_dir))
      run(cmd)
  markDone('compute_yahoo_egain_feature')

if logDo('compute_eod_volatility'):
  for k in VOLATILITY_K_LIST:
    volatility_dir = '%s_%d' % (EOD_VOLATILITY_PREFIX, k)
    util.maybeMakeDir(volatility_dir)
    cmd = ('%s/compute_volatility.py --price_dir=%s --k=%d '
           '--volatility_dir=%s' % (
        CODE_DIR, EOD_ADJPRICE_DIR, k, volatility_dir))
    run(cmd)
  markDone('compute_eod_volatility')

if logDo('compute_yahoo_volatility'):
  for k in VOLATILITY_K_LIST:
    volatility_dir = '%s_%d' % (YAHOO_VOLATILITY_PREFIX, k)
    util.maybeMakeDir(volatility_dir)
    cmd = ('%s/compute_volatility.py --price_dir=%s --k=%d '
           '--volatility_dir=%s' % (
        CODE_DIR, YAHOO_ADJPRICE_DIR, k, volatility_dir))
    run(cmd)
  markDone('compute_yahoo_volatility')

if logDo('compute_eod_volatility_perc'):
  for k in VOLATILITY_K_LIST:
    input_dir = '%s_%d' % (EOD_VOLATILITY_PREFIX, k)
    output_dir = '%s_%d' % (EOD_VOLATILITY_PERC_PREFIX, k)
    util.maybeMakeDir(output_dir)
    cmd = ('%s/compute_rank_perc.py --input_dir=%s '
           '--output_dir=%s' % (CODE_DIR, input_dir, output_dir))
    run(cmd)
  markDone('compute_eod_volatility_perc')

if logDo('compute_yahoo_volatility_perc'):
  for k in VOLATILITY_K_LIST:
    input_dir = '%s_%d' % (YAHOO_VOLATILITY_PREFIX, k)
    output_dir = '%s_%d' % (YAHOO_VOLATILITY_PERC_PREFIX, k)
    util.maybeMakeDir(output_dir)
    cmd = ('%s/compute_rank_perc.py --input_dir=%s '
           '--output_dir=%s' % (CODE_DIR, input_dir, output_dir))
    run(cmd)
  markDone('compute_yahoo_volatility_perc')

for experiment in EXPERIMENTS:
  config_file = '%s/%s.json' % (CONFIG_DIR, experiment)
  cmd = '%s/run_experiment.py --config=%s' % (CODE_DIR, config_file)
  run(cmd)

