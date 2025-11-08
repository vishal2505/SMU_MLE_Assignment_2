import os
from datetime import datetime
import pyspark.sql.functions as F

from pyspark.sql.functions import col


def process_bronze_loan_daily_table(snapshot_date_str, bronze_directory, spark):

    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    csv_file_path = "data/lms_loan_daily.csv"

    bronze_loan_daily = bronze_directory + "bronze_loan_daily/"
    os.makedirs(bronze_loan_daily, exist_ok=True)

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_loan_daily + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df

def process_bronze_clickstream_table(snapshot_date_str, bronze_directory, spark):

    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    csv_file_path = "data/feature_clickstream.csv"
    bronze_clickstream = bronze_directory + "bronze_clickstream/"
    os.makedirs(bronze_clickstream, exist_ok=True)

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df

def process_bronze_attributes_table(snapshot_date_str, bronze_directory, spark):

    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    csv_file_path = "data/features_attributes.csv"
    bronze_attributes = bronze_directory + "bronze_attributes/"
    os.makedirs(bronze_attributes, exist_ok=True)

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df

def process_bronze_financials_table(snapshot_date_str, bronze_directory, spark):

    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    csv_file_path = "data/features_financials.csv"
    bronze_financials = bronze_directory + "bronze_financials/"
    os.makedirs(bronze_financials, exist_ok=True)

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df

def process_all_bronze_tables(snapshot_date_str, bronze_directory, spark):
    
    print(f"\n{'='*70}")
    print(f"Processing Bronze Tables for snapshot date: {snapshot_date_str}")
    print(f"{'='*70}\n")
    
    # connect to source back end - IRL connect to back end source system
    print("1. Processing Loan Daily Table...")
    loan_daily_df = process_bronze_loan_daily_table(snapshot_date_str, bronze_directory, spark)
    print("Loan Daily completed\n")

    print("2. Processing Clickstream Table...")
    clickstream_df = process_bronze_clickstream_table(snapshot_date_str, bronze_directory, spark)
    print("Clickstream completed\n")

    print("3. Processing Attributes Table...")
    attributes_df = process_bronze_attributes_table(snapshot_date_str, bronze_directory, spark)
    print("Attributes completed\n")

    print("4. Processing Financials Table...")  
    financials_df = process_bronze_financials_table(snapshot_date_str, bronze_directory, spark)
    print("Financials completed\n")
    
    print(f"{'='*70}")
    print(f"Bronze Tables Processing Complete for {snapshot_date_str}")
    print(f"{'='*70}\n")

    return (loan_daily_df, clickstream_df, attributes_df, financials_df)
