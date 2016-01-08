import socket

# Local or remote.
HOST = socket.gethostname()
assert HOST == 'lnyang-mn1' or HOST == 'lnyang-ld1'
if HOST == 'lnyang-mn1':
  HOST_DIR = '/Users/lnyang/lab/qd2'
else:
  assert HOST == 'lnyang-ld1'
  HOST_DIR = '/home/lnyang/lab/qd2'

###############
## Constants ##
###############

RUN_ID = '20160101'
EXPERIMENTS = [
]

MARKETS = ['R3000', 'SP500']

TEST = False
DRY_RUN = False

# For features, we look at many time windows, and we do not
# enforce any minimum raw price.
# Price/volume features have been disabled.
# GAIN_K_LIST is also used for computing market labels;
# it is emptied so that no useless data is produced.
GAIN_K_LIST = []  # [1, 2, 3, 6, 9, 12, 15, 18, 21, 24,
                  #  27, 30, 33, 36, 39, 42, 45, 48]
PRICE_K_LIST = [1, 2, 3, 6, 9, 12, 15, 18, 21, 24,
                27, 30, 33, 36, 39, 42, 45, 48]
LOGPRICE_K_LIST = [1, 2, 3, 6, 9, 12, 15, 18, 21, 24,
                   27, 30, 33, 36, 39, 42, 45, 48]
ADJPRICE_K_LIST = [1, 2, 3, 6, 9, 12, 15, 18, 21, 24,
                   27, 30, 33, 36, 39, 42, 45, 48]
LOGADJPRICE_K_LIST = [1, 2, 3, 6, 9, 12, 15, 18, 21, 24,
                      27, 30, 33, 36, 39, 42, 45, 48]
LOGADJVOLUME_K_LIST = [1, 2, 3, 6, 9, 12, 15, 18, 21, 24,
                       27, 30, 33, 36, 39, 42, 45, 48]

# Time windows for computing volatility.
FILTER_VOLATILITY_K = 24  # This must in the list below.
VOLATILITY_K_LIST = [24, 48]

# For labels, we only predict a 12-month window, and we enforce
# a minimum raw price and/or membership for all transactions.
PREDICTION_WINDOW = 12
MEMBERSHIP = 'SP500'

# [[max_look, max_pick, max_hold] ...]
# if max_look < 0: no limit
# if max_pick < 0: pick bottom-ranked stocks (to simulate short)
# if max_hold < 0: no limit
TRADE_CONFIGS = [
    # long
    [-1, 5, 1],
    [-1, 5, 2],
    [-1, 5, 3],
    [-1, 5, 4],
    [-1, 5, 5],
    [-1, 5, 10],
    [-1, 5, -1],
]

CODE_DIR = '%s/qd2' % HOST_DIR

if TEST:
  RUN_DIR = '%s/testdata' % CODE_DIR
  EVAL_PERCS = [10, 100]
  KS = [1, 3, 5, 0, -5, -3, -1]
  BUCKETS_LIST = [2]
  TOPK = 3
else:
  RUN_DIR = '%s/data/runs/%s' % (HOST_DIR, RUN_ID)
  EVAL_PERCS = [1, 10, 100]
  KS = [1, 3, 5, 10, 30, 50, 100, 0, -100, -50, -30, -10, -5, -3, -1]
  BUCKETS_LIST = [10, 30, 100]
  TOPK = 5

SYMBOL_DIR = '%s/symbols' % RUN_DIR

RAW_DIR = '%s/raw' % RUN_DIR
RAW_SF1_DIR = '%s/sf1' % RAW_DIR
RAW_EOD_DIR = '%s/eod' % RAW_DIR
RAW_YAHOO_DIR = '%s/yahoo' % RAW_DIR
RAW_MISC_DIR = '%s/misc' % RAW_DIR

RAW_SF1_FILE = '%s/sf1.csv' % RAW_SF1_DIR
SF1_INDICATOR_FILE = '%s/indicators.txt' % RAW_SF1_DIR
SF1_INFO_FILE = '%s/tickers.txt' % RAW_SF1_DIR
RAW_EOD_FILE = '%s/eod.csv' % RAW_EOD_DIR

TICKER_DIR = '%s/tickers' % RUN_DIR
SF1_TICKER_FILE = '%s/sf1_tickers' % TICKER_DIR
EOD_TICKER_FILE = '%s/eod_tickers' % TICKER_DIR
YAHOO_TICKER_FILE = '%s/yahoo_tickers' % TICKER_DIR

YAHOO_SF1_DIR = '%s/sf1' % RAW_YAHOO_DIR
YAHOO_MARKET_DIR = '%s/market' % RAW_YAHOO_DIR

SF1_DIR = '%s/sf1' % RUN_DIR
SF1_RAW_DIR = '%s/raw' % SF1_DIR
SF1_PROCESSED_DIR = '%s/processed' % SF1_DIR

EOD_DIR = '%s/eod' % RUN_DIR
EOD_RAW_DIR = '%s/raw' % EOD_DIR
EOD_PROCESSED_DIR = '%s/processed' % EOD_DIR
EOD_PRICE_DIR = '%s/price' % EOD_DIR
EOD_ADJPRICE_DIR = '%s/adjprice' % EOD_DIR
EOD_LOGPRICE_DIR = '%s/logprice' % EOD_DIR
EOD_LOGADJPRICE_DIR = '%s/logadjprice' % EOD_DIR
EOD_LOGADJVOLUME_DIR = '%s/logadjvolume' % EOD_DIR
EOD_GAIN_DIR = '%s/gain' % EOD_DIR
EOD_EGAIN_DIR = '%s/egain' % EOD_DIR

YAHOO_DIR = '%s/yahoo' % RUN_DIR
YAHOO_PROCESSED_DIR = '%s/processed' % YAHOO_DIR
YAHOO_PRICE_DIR = '%s/price' % YAHOO_DIR
YAHOO_ADJPRICE_DIR = '%s/adjprice' % YAHOO_DIR
YAHOO_LOGPRICE_DIR = '%s/logprice' % YAHOO_DIR
YAHOO_LOGADJPRICE_DIR = '%s/logadjprice' % YAHOO_DIR
YAHOO_LOGADJVOLUME_DIR = '%s/logadjvolume' % YAHOO_DIR
YAHOO_GAIN_DIR = '%s/gain' % YAHOO_DIR
YAHOO_EGAIN_DIR = '%s/egain' % YAHOO_DIR

YAHOO_TRADING_DAY_FILE = '%s/trading_days' % YAHOO_DIR
YAHOO_HOLE_DIR = '%s/hole' % YAHOO_DIR

MARKET_DIR = '%s/market' % RUN_DIR
MARKET_PROCESSED_DIR = '%s/processed' % MARKET_DIR
MARKET_ADJPRICE_DIR = '%s/adjprice' % MARKET_DIR
MARKET_GAIN_DIR = '%s/gain' % MARKET_DIR

FEATURE_DIR = '%s/features' % RUN_DIR
FEATURE_INFO_DIR = '%s/feature_info' % RUN_DIR
FEATURE_LIST_DIR = '%s/feature_lists' % RUN_DIR

EOD_VOLATILITY_PREFIX = '%s/eod_volatility_' % FEATURE_DIR
EOD_VOLATILITY_PERC_PREFIX = '%s/eod_volatility_perc_' % FEATURE_DIR
YAHOO_VOLATILITY_PREFIX = '%s/yahoo_volatility_' % FEATURE_DIR
YAHOO_VOLATILITY_PERC_PREFIX = '%s/yahoo_volatility_perc_' % FEATURE_DIR

MISC_DIR = '%s/misc' % RUN_DIR
FEATURE_STATS_FILE = '%s/feature_stats.tsv' % MISC_DIR
SECTOR_MAP_FILE = '%s/sector_map.tsv' % MISC_DIR
INDUSTRY_MAP_FILE = '%s/industry_map.tsv' % MISC_DIR
SECTOR_STATS_FILE = '%s/sector_stats' % MISC_DIR
INDUSTRY_STATS_FILE = '%s/industry_stats' % MISC_DIR
MEMBERSHIP_FILE = '%s/%s-membership' % (MISC_DIR, MEMBERSHIP)

EOD_GAIN_LABEL_DIR = '%s/gain_label/%d' % (EOD_DIR, PREDICTION_WINDOW)
YAHOO_GAIN_LABEL_DIR = '%s/gain_label/%d' % (YAHOO_DIR, PREDICTION_WINDOW)
EOD_EGAIN_LABEL_DIR = '%s/egain_label/%d' % (EOD_DIR, PREDICTION_WINDOW)
YAHOO_EGAIN_LABEL_DIR = '%s/egain_label/%d' % (YAHOO_DIR, PREDICTION_WINDOW)

CONFIG_DIR = '%s/configs' % RUN_DIR
EXPERIMENT_BASE_DIR = '%s/experiments' % RUN_DIR

