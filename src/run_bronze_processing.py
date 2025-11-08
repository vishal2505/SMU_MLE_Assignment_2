#!/usr/bin/env python
"""
Wrapper script for Bronze layer processing
Called by Airflow DAG
"""

import sys
import argparse
from pathlib import Path
from pyspark.sql import SparkSession

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline_config.config import BRONZE_PATH, SPARK_MASTER, SPARK_DRIVER_MEMORY, SPARK_APP_NAME
from src.data_processing_bronze_table import process_all_bronze_tables


def main():
    parser = argparse.ArgumentParser(description='Process Bronze layer tables')
    parser.add_argument('--snapshot_date', type=str, required=True, help='Snapshot date (YYYY-MM-DD)')
    args = parser.parse_args()
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName(f"{SPARK_APP_NAME}_Bronze") \
        .master(SPARK_MASTER) \
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
        .getOrCreate()
    
    try:
        # Process bronze tables
        process_all_bronze_tables(
            args.snapshot_date,
            str(BRONZE_PATH) + "/",
            spark
        )
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
