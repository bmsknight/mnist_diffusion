import math
import os

# Paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/preprocessed"
OUTPUT_DIR = "output"
UTIL_DIR = "utilismart_october"

UTILISMART_DATA_FILE_NAME = "utilismart_dataset2.csv"
KAGGLE_DATA_FILE_NAME = "kaggle/load_history.csv"
KAGGLE_BENCHMARK_FILE_NAME = "kaggle/benchmark.csv"
CONFIG_PATH = "config/config.yaml"
DAE_CONFIG_PATH = "config/dae_config.yml"
OUT_FILE_NAME_TEMPLATE = "{time}_{model}_{dataset}_{stream}"
OUTPUT_TIME_FORMAT = "%d-%m-%Y-%H-%M-%S"
TEMPORARY_MODEL_SAVE_PATH = "models/temp_best_model.pth"
TEMPORARY_MODEL_SAVE_PATH_WITH_ID = "models/temp_best_model_run_{runid}.pth"

UTILISMART_RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, UTILISMART_DATA_FILE_NAME)
UTILISMART_PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, UTILISMART_DATA_FILE_NAME)
KAGGLE_RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, KAGGLE_DATA_FILE_NAME)
KAGGLE_RAW_BENCHMARK_PATH = os.path.join(RAW_DATA_DIR, KAGGLE_BENCHMARK_FILE_NAME)
KAGGLE_PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, KAGGLE_DATA_FILE_NAME)
OUT_FILE_PATH_TEMPLATE = os.path.join(OUTPUT_DIR, OUT_FILE_NAME_TEMPLATE)

# Optuna constants
OPTUNA_DEFAULT_DB = "sqlite:///output/optuna.db"
OPTUNA_DEFAULT_STUDY_NAME = "hyper_search_kaggle_usad_0"

# Utilismart Dataset fields and categories
METER_ID = "SENSORID"

READING_TYPE = "CHANTYPE"
REGISTER_READING = 1
INTERVAL_READING = 2

READING_VALUE = "VAL"

READING_TIMESTAMP = "READTS"
TIMESTAMP_FORMAT = "%d-%b-%y %I.%M.%S.%f %p"

READING_INTERVAL = "INTV"

READING_UNIT = "UOM"
KWH = 6

READING_STATE = "STATE"
ACTUAL_READING = 3
IMPUTED_READING = 5

SUPPLY_DIRECTION = "DIR"
DIR_CONSUMED = 1
DIR_GENERATED = 2

# training related constants
TEST_SPLIT_FRAC = 0.2
NUM_WORKERS = 3

# Utilismart Hours
HOUR_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

# Kaggle Dataset fields and categories
K_METER_ID = "zone_id"
K_HOUR_LIST = ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10", "h11", "h12",
               "h13", "h14", "h15", "h16", "h17", "h18", "h19", "h20", "h21", "h22", "h23", "h24"]

K_YEAR = "year"
K_MONTH = "month"
K_DAY = "day"
K_HOUR = "HOUR"
K_READING_VALUE = "VAL"
K_READING_TIMESTAMP = "READTS"

K_READING_STATE = "IMPUTED"
K_IMPUTED_READING = True
K_ACTUAL_READING = False

# Evaluation dataframe columns
EVAL_TARGET_COLUMN = "actuals"
EVAL_INPUT_COLUMN = "actuals_with_missing_values"
EVAL_PREDICTION_COLUMN = "predictions"
EVAL_IS_MISSING_COLUMN = "missing_value_filter"

# Sine wave constants for debugging
SIN_OMEGA = 2 * math.pi / 360
SIN_AMPLITUDE = 1
