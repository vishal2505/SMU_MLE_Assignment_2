import os
import glob
from datetime import datetime
import pyspark

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


def main():
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("Credit_Default_Data_Pipeline") \
        .master("local[*]") \
        .getOrCreate()

    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # set up config
    start_date_str = "2023-01-01"
    end_date_str = "2024-12-01"

    # generate list of dates to process
    def generate_first_of_month_dates(start_date_str, end_date_str):
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        
        first_of_month_dates = []
        current_date = datetime(start_date.year, start_date.month, 1)

        while current_date <= end_date:
            first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
            
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)

        return first_of_month_dates

    dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
    print(f"\n{'='*70}")
    print(f"Credit Default Prediction - Data Pipeline")
    print(f"{'='*70}")
    print(f"Processing {len(dates_str_lst)} snapshots from {start_date_str} to {end_date_str}")
    print(f"{'='*70}\n")

    # create bronze datalake
    bronze_directory = "datamart/bronze/"
    os.makedirs(bronze_directory, exist_ok=True)

    # run bronze backfill
    print(f"\n{'#'*70}")
    print(f"# BRONZE LAYER PROCESSING")
    print(f"{'#'*70}\n")
    for idx, date_str in enumerate(dates_str_lst, 1):
        print(f"[{idx}/{len(dates_str_lst)}] Processing {date_str}...")
        utils.data_processing_bronze_table.process_all_bronze_tables(date_str, bronze_directory, spark)

    # create silver datalake
    silver_directory = "datamart/silver/"
    os.makedirs(silver_directory, exist_ok=True)

    # run silver backfill
    print(f"\n{'#'*70}")
    print(f"# SILVER LAYER PROCESSING")
    print(f"{'#'*70}\n")
    for idx, date_str in enumerate(dates_str_lst, 1):
        print(f"[{idx}/{len(dates_str_lst)}] Processing {date_str}...")
        utils.data_processing_silver_table.process_all_silver_tables(date_str, bronze_directory, silver_directory, spark)

    # create gold directories
    gold_directory = "datamart/gold/"
    os.makedirs(gold_directory, exist_ok=True)

    # run gold backfill
    print(f"\n{'#'*70}")
    print(f"# GOLD LAYER PROCESSING")
    print(f"{'#'*70}\n")
    for idx, date_str in enumerate(dates_str_lst, 1):
        print(f"[{idx}/{len(dates_str_lst)}] Processing {date_str}...")
        utils.data_processing_gold_table.process_all_gold_tables(
            date_str, silver_directory, gold_directory, spark, dpd=30, mob=6
        )

    # Read and display feature store
    print(f"\n{'='*70}")
    print(f"FEATURE STORE SUMMARY")
    print(f"{'='*70}")
    feature_store_directory = os.path.join(gold_directory, "feature_store")
    feature_files = [os.path.join(feature_store_directory, os.path.basename(f)) 
                     for f in glob.glob(os.path.join(feature_store_directory, '*.parquet'))]
    
    if feature_files:
        df_features = spark.read.parquet(*feature_files)
        print(f"Feature Store row count: {df_features.count()}")
        print(f"Feature Store columns: {len(df_features.columns)}")
        print(f"\nColumn names: {df_features.columns}")
        print("\nFeature Store Schema:")
        df_features.printSchema()
        print("\nSample Data:")
        df_features.show(5, truncate=False)
    else:
        print("No feature store files found!")

    # Read and display label store
    print(f"\n{'='*70}")
    print(f"LABEL STORE SUMMARY")
    print(f"{'='*70}")
    label_store_directory = os.path.join(gold_directory, "label_store")
    label_files = [os.path.join(label_store_directory, os.path.basename(f)) 
                   for f in glob.glob(os.path.join(label_store_directory, '*.parquet'))]
    
    if label_files:
        df_labels = spark.read.parquet(*label_files)
        print(f"Label Store row count: {df_labels.count()}")
        print(f"\nLabel distribution:")
        df_labels.groupBy("label").count().orderBy("label").show()
        print("\nSample Data:")
        df_labels.show(5, truncate=False)
    else:
        print("No label store files found!")

    # Stop Spark
    spark.stop()


if __name__ == "__main__":
    main()



