#!/usr/bin/env python
"""
Wrapper script for Gold layer processing
Called by Airflow DAG
"""

import sys
import argparse
from pathlib import Path
from pyspark.sql import SparkSession

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline_config.config import (
    SILVER_PATH, GOLD_PATH, SPARK_MASTER, SPARK_DRIVER_MEMORY, 
    SPARK_APP_NAME, DPD_THRESHOLD, LABEL_MOB, FEATURE_MOB
)
from src.data_processing_gold_table import process_all_gold_tables


def main():
    parser = argparse.ArgumentParser(description='Process Gold layer tables')
    parser.add_argument('--snapshot_date', type=str, required=True, help='Snapshot date (YYYY-MM-DD)')
    args = parser.parse_args()
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName(f"{SPARK_APP_NAME}_Gold") \
        .master(SPARK_MASTER) \
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
        .getOrCreate()
    
    try:
        # Process gold tables
        process_all_gold_tables(
            args.snapshot_date,
            str(SILVER_PATH) + "/",
            str(GOLD_PATH) + "/",
            spark,
            dpd=DPD_THRESHOLD,
            mob=LABEL_MOB,
            feature_mob=FEATURE_MOB
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
