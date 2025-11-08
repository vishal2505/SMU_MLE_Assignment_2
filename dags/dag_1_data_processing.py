"""
DAG 1: Data Processing Pipeline
Processes Bronze → Silver → Gold layers
Runs monthly from 2023-01-01 to 2024-12-01
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
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
    'dag_1_data_processing',
    default_args=default_args,
    description='Data processing pipeline: Bronze → Silver → Gold',
    schedule_interval='0 0 1 * *',  # Monthly on the 1st
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
    max_active_runs=1,
    tags=['data-processing', 'bronze', 'silver', 'gold'],
)

# ============================================================================
# DATA PROCESSING TASKS
# ============================================================================

start_pipeline = DummyOperator(
    task_id='start_data_processing',
    dag=dag,
)

# Bronze Layer
bronze_layer_start = DummyOperator(
    task_id='bronze_layer_start',
    dag=dag,
)

process_bronze_tables = BashOperator(
    task_id='process_bronze_tables',
    bash_command="""
    cd /opt/airflow && \
    python src/run_bronze_processing.py --snapshot_date {{ ds }}
    """,
    dag=dag,
)

bronze_layer_complete = DummyOperator(
    task_id='bronze_layer_complete',
    dag=dag,
)

# Silver Layer
silver_layer_start = DummyOperator(
    task_id='silver_layer_start',
    dag=dag,
)

process_silver_tables = BashOperator(
    task_id='process_silver_tables',
    bash_command="""
    cd /opt/airflow && \
    python src/run_silver_processing.py --snapshot_date {{ ds }}
    """,
    dag=dag,
)

silver_layer_complete = DummyOperator(
    task_id='silver_layer_complete',
    dag=dag,
)

# Gold Layer
gold_layer_start = DummyOperator(
    task_id='gold_layer_start',
    dag=dag,
)

process_gold_tables = BashOperator(
    task_id='process_gold_tables',
    bash_command="""
    cd /opt/airflow && \
    python src/run_gold_processing.py --snapshot_date {{ ds }}
    """,
    dag=dag,
)

gold_layer_complete = DummyOperator(
    task_id='gold_layer_complete',
    dag=dag,
)

# Pipeline complete
data_processing_complete = DummyOperator(
    task_id='data_processing_complete',
    dag=dag,
)

# ============================================================================
# TASK DEPENDENCIES
# ============================================================================

start_pipeline >> bronze_layer_start
bronze_layer_start >> process_bronze_tables >> bronze_layer_complete

bronze_layer_complete >> silver_layer_start
silver_layer_start >> process_silver_tables >> silver_layer_complete

silver_layer_complete >> gold_layer_start
gold_layer_start >> process_gold_tables >> gold_layer_complete

gold_layer_complete >> data_processing_complete
