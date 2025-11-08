"""
DAG 2: Model Training Pipeline
Trains ML models after all data processing is complete
Runs only once on 2024-12-01
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.dummy import DummyOperator
import sys

# Add project root to path
sys.path.insert(0, '/opt/airflow')

# Import configuration directly
import importlib.util
spec = importlib.util.spec_from_file_location("pipeline_config", "/opt/airflow/pipeline_config/config.py")
pipeline_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_config)

AIRFLOW_RETRIES = pipeline_config.AIRFLOW_RETRIES
AIRFLOW_RETRY_DELAY_MINUTES = pipeline_config.AIRFLOW_RETRY_DELAY_MINUTES
MODEL_TRAINING_TRIGGER_DATE = pipeline_config.MODEL_TRAINING_TRIGGER_DATE

# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': AIRFLOW_RETRIES,
    'retry_delay': timedelta(minutes=AIRFLOW_RETRY_DELAY_MINUTES),
}

# DAG definition
dag = DAG(
    'dag_2_model_training',
    default_args=default_args,
    description='Model training pipeline: Runs once when data through OOT end is available',
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2024, 12, 1),
    catchup=True,
    max_active_runs=1,
    tags=['model-training', 'ml'],
)

# ============================================================================
# CHECK DATA AVAILABILITY
# ============================================================================

def check_all_data_processed(**context):
    """Return True if all required feature snapshots and their aligned LABEL_MOB month label snapshots exist for train/val/test windows.

    Strict four-window enforcement: ensures validation and test have labels available after LABEL_MOB alignment.
    """
    from pathlib import Path
    from dateutil.relativedelta import relativedelta
    import datetime as pydt

    cfg = pipeline_config

    def to_dt(s: str) -> pydt.datetime:
        return pydt.datetime.fromisoformat(s)

    def month_floor(dt: pydt.datetime) -> pydt.datetime:
        return pydt.datetime(dt.year, dt.month, 1)

    def month_range(start_dt: pydt.datetime, end_dt: pydt.datetime):
        cur = month_floor(start_dt)
        end = month_floor(end_dt)
        while cur <= end:
            yield cur
            cur += relativedelta(months=1)

    label_offset = getattr(cfg, 'LABEL_MOB', 6)

    # Build the set of required feature months across train/val/test
    req_feature_months = []
    for split in ('train', 'validation', 'test'):
        s = to_dt(cfg.TEMPORAL_SPLITS[split]['start_date'])
        e = to_dt(cfg.TEMPORAL_SPLITS[split]['end_date'])
        req_feature_months.extend(list(month_range(s, e)))

    gold_path = Path("/opt/airflow/datamart/gold")
    feature_store_path = gold_path / "feature_store"
    label_store_path = gold_path / "label_store"

    missing = []
    for cur in req_feature_months:
        tag = cur.strftime("%Y_%m_%d")
        # Feature snapshot for this month must exist
        f_path = feature_store_path / f"feature_store_{tag}.parquet"
        if not f_path.exists():
            missing.append(str(f_path))
        # Aligned label snapshot LABEL_MOB months after this feature month must exist
        label_month = cur + relativedelta(months=label_offset)
        l_tag = label_month.strftime("%Y_%m_%d")
        l_path = label_store_path / f"label_store_{l_tag}.parquet"
        if not l_path.exists():
            missing.append(str(l_path))

    if missing:
        print(f"✗ Training readiness check failed. Missing {len(missing)} files.")
        for m in missing[:20]:
            print("  -", m)
        if len(missing) > 20:
            print(f"  ... and {len(missing)-20} more")
        return False

    start_date = min(req_feature_months).strftime('%Y-%m-%d')
    end_date = max(req_feature_months).strftime('%Y-%m-%d')
    print("✓ Training readiness confirmed. All required feature and aligned label snapshots present.")
    print(f"  Feature range: {start_date} → {end_date}")
    print(f"  Label offset (months): {label_offset}")
    print(f"  Feature store dir: {feature_store_path}")
    print(f"  Label store dir:   {label_store_path}")
    return True


# ============================================================================
# MODEL TRAINING TASKS
# ============================================================================

start_training_pipeline = DummyOperator(task_id='start_training_pipeline', dag=dag)

# Short circuit if readiness not met (task downstream won't run)
check_data_availability = ShortCircuitOperator(
    task_id='check_data_availability',
    python_callable=check_all_data_processed,
    dag=dag,
)

# Train models using temporal windows
train_models = BashOperator(
    task_id='train_models',
    bash_command="""
    cd /opt/airflow && \
    python src/model_training.py --use_temporal_splits
    """,
    dag=dag,
)

model_training_complete = DummyOperator(
    task_id='model_training_complete',
    dag=dag,
)

# ============================================================================
# TASK DEPENDENCIES
# ============================================================================

start_training_pipeline >> check_data_availability >> train_models >> model_training_complete
