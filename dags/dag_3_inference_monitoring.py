"""
DAG 3: Model Inference and Monitoring Pipeline

Range-enabled version:
    - Inference: generates monthly predictions from DATA_PROCESSING_START_DATE through DATA_PROCESSING_END_DATE.
    - Monitoring: evaluates monthly predictions from DATA_PROCESSING_START_DATE through OOT_START_DATE
        (latest snapshot date whose +LABEL_MOB labels exist). Labels exist up to DATA_PROCESSING_END_DATE
        (2024-12-01), with LABEL_MOB=6 ⇒ last monitorable snapshot date is 2024-06-01.

If you prefer to restrict inference only to monitorable months, change INFER_END_DATE to OOT_START_DATE.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
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
OOT_START_DATE = pipeline_config.TEMPORAL_SPLITS['oot']['start_date']
OOT_END_DATE = pipeline_config.TEMPORAL_SPLITS['oot']['end_date']

INFER_START_DATE = pipeline_config.INFER_START_DATE
INFER_END_DATE = pipeline_config.INFER_END_DATE
MONITOR_START_DATE = pipeline_config.MONITOR_START_DATE
MONITOR_END_DATE = pipeline_config.MONITOR_END_DATE

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
    'dag_3_inference_monitoring',
    default_args=default_args,
    description='Model inference and monitoring on OOT data',
    schedule_interval=None,  # Manual trigger after training completes
    start_date=datetime(2024, 12, 1),
    catchup=True,
    max_active_runs=1,
    tags=['inference', 'monitoring', 'oot'],
)

# ============================================================================
# SIMPLE OOT DATA CHECK (optional, logs only)
# ============================================================================

def check_data_availability(**context):
    """Check if OOT data is available"""
    import os
    from pathlib import Path
    
    gold_path = Path("/opt/airflow/datamart/gold")
    oot_date_str = OOT_START_DATE.replace('-', '_')
    
    feature_file = gold_path / "feature_store" / f"feature_store_{oot_date_str}.parquet"
    label_file = gold_path / "label_store" / f"label_store_{oot_date_str}.parquet"
    
    if feature_file.exists() and label_file.exists():
        print(f"✓ OOT data available")
        print(f"  Features: {feature_file}")
        print(f"  Labels: {label_file}")
        print(f"  OOT date: {OOT_START_DATE}")
        return True
    else:
        missing = []
        if not feature_file.exists():
            missing.append(f"  Features: {feature_file}")
        if not label_file.exists():
            missing.append(f"  Labels: {label_file}")
        
        print(f"✗ Missing OOT data files:")
        for m in missing:
            print(m)
        return False


# ============================================================================
# INFERENCE AND MONITORING TASKS
# ============================================================================

start_inference_pipeline = DummyOperator(task_id='start_inference_pipeline', dag=dag)

# Check if OOT data is available
check_data_availability = PythonOperator(task_id='check_data_availability', python_callable=check_data_availability, provide_context=True, dag=dag)

# Run inference across full date range
run_inference_range = BashOperator(
    task_id='run_inference_range',
    bash_command=f"""
    cd /opt/airflow && \
    python src/model_inference.py --start_date {INFER_START_DATE} --end_date {INFER_END_DATE}
    """,
    dag=dag,
)

# Run monitoring across monitorable date range (labels aligned +6M available)
run_monitoring_range = BashOperator(
    task_id='run_monitoring_range',
    bash_command=f"""
    cd /opt/airflow && \
    python src/model_monitoring.py --start_date {MONITOR_START_DATE} --end_date {MONITOR_END_DATE}
    """,
    dag=dag,
)

# Generate visualizations
generate_visualizations = BashOperator(
    task_id='generate_visualizations',
    bash_command="""
    cd /opt/airflow && \
    python src/visualization.py
    """,
    dag=dag,
)

pipeline_complete = DummyOperator(
    task_id='pipeline_complete',
    dag=dag,
)

# ============================================================================
# TASK DEPENDENCIES
# ============================================================================

start_inference_pipeline >> check_data_availability >> run_inference_range >> run_monitoring_range >> generate_visualizations >> pipeline_complete
