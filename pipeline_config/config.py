"""
Configuration File - Pipeline Constants
Centralizes all configuration parameters
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data"
BRONZE_PATH = PROJECT_ROOT / "datamart" / "bronze"
SILVER_PATH = PROJECT_ROOT / "datamart" / "silver"
GOLD_PATH = PROJECT_ROOT / "datamart" / "gold"
MODEL_STORE_PATH = PROJECT_ROOT / "model_store"
RESULTS_PATH = PROJECT_ROOT / "results"

# Create directories if they don't exist
for path in [DATA_PATH, BRONZE_PATH, SILVER_PATH, GOLD_PATH, MODEL_STORE_PATH, RESULTS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Gold layer parameters
FEATURE_MOB = 0  # Features at application time (MOB=0)
LABEL_MOB = 6    # Labels at 6 months (MOB=6)
DPD_THRESHOLD = 30  # Days past due threshold for default definition

# Temporal Window Configuration
TEMPORAL_WINDOW_MODE = "absolute"  # "absolute" or "relative"
TEMPORAL_SPLITS = {
    "train": {
        "start_date": "2023-01-01",
        "end_date": "2023-12-01"
    },
    "validation": {
        "start_date": "2024-01-01",
        "end_date": "2024-03-01"
    },
    "test": {
        "start_date": "2024-04-01",
        "end_date": "2024-05-01"
    },
    "oot": {
        "start_date": "2024-06-01",
        "end_date": "2024-06-01"
    }
}

# Data processing date range (for backfill)
DATA_PROCESSING_START_DATE = "2023-01-01"
DATA_PROCESSING_END_DATE = "2024-12-01"

# Model training trigger date (only train when all data is available)
MODEL_TRAINING_TRIGGER_DATE = "2024-12-01"

INFER_START_DATE = "2024-04-01"
INFER_END_DATE = "2024-06-01"
MONITOR_START_DATE = "2024-04-01"
MONITOR_END_DATE = "2024-06-01"

# Model training parameters
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.
RANDOM_STATE = 42
CV_FOLDS = 5

# Model configurations
MODELS = {
    'LogisticRegression': {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': RANDOM_STATE,
        'solver': 'lbfgs',
        'class_weight': 'balanced'
    },
    'RandomForest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE,
        'class_weight': 'balanced',
        'n_jobs': -1
    },
    'XGBoost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
}

# Feature configuration - Features available at application time (MOB=0)
# These features do NOT cause temporal leakage
FEATURE_COLUMNS = [
    # Loan characteristics at application
    'tenure',
    'loan_amt',
    
    # Customer demographics
    'customer_age',
    
    # Financial features
    'Annual_Income',
    'Monthly_Inhand_Salary',
    'Num_Bank_Accounts',
    'Num_Credit_Card',
    'Interest_Rate',
    'Num_of_Loan',
    'Num_of_Delayed_Payment',
    'Outstanding_Debt',
    'Credit_Utilization_Ratio',
    'Total_EMI_per_month',
    'Amount_invested_monthly',
    'Monthly_Balance',
    'debt_to_income_ratio',
    
    # Clickstream features
    'fe_1', 'fe_2', 'fe_3', 'fe_4', 'fe_5',
    'fe_6', 'fe_7', 'fe_8', 'fe_9', 'fe_10',
    'fe_11', 'fe_12', 'fe_13', 'fe_14', 'fe_15',
    'fe_16', 'fe_17', 'fe_18', 'fe_19', 'fe_20'
]

# Monitoring thresholds
MONITORING_THRESHOLDS = {
    'auc_roc_min': 0.70,
    'precision_min': 0.60,
    'recall_min': 0.50,
    'f1_score_min': 0.55,
    'psi_warning': 0.1,
    'psi_critical': 0.2,
    'performance_degradation_threshold': 0.05  # 5% drop triggers retraining
}

# Monitoring metrics to track
MONITORING_METRICS = [
    'auc_roc',
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'log_loss',
    'psi'
]

# Model selection criteria
MODEL_SELECTION_METRIC = 'auc_roc'  # Primary metric for selecting best model
MODEL_SELECTION_MODE = 'max'  # 'max' or 'min'

# Airflow parameters
AIRFLOW_SCHEDULE = '0 0 1 * *'  # Monthly at midnight on the 1st
AIRFLOW_EMAIL = ['ml-team@company.com']
AIRFLOW_RETRIES = 2
AIRFLOW_RETRY_DELAY_MINUTES = 5
AIRFLOW_EXECUTION_TIMEOUT_HOURS = 2

# Spark configuration
SPARK_MASTER = "local[*]"
SPARK_DRIVER_MEMORY = "4g"
SPARK_APP_NAME = "MLPipeline"

# File formats
PARQUET_COMPRESSION = 'snappy'
PARQUET_ENGINE = 'pyarrow'

# Training data windows (in months)
TRAINING_WINDOW_MONTHS = 12
VALIDATION_WINDOW_MONTHS = 2
TEST_WINDOW_MONTHS = 2

# Model versioning
MODEL_CONFIG_FILE = "model_config.json"
MODEL_EVALUATION_FILE = "model_evaluation.json"
MODEL_MONITORING_FILE = "model_monitoring.json"