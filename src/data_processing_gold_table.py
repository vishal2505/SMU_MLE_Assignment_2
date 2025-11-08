import os
import glob
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse
from pathlib import Path

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

# Import pipeline config constants for consistency
sys.path.append(str(Path(__file__).parent.parent))
from pipeline_config.config import LABEL_MOB, DPD_THRESHOLD, FEATURE_MOB


def process_label_store(snapshot_date_str, silver_directory, gold_directory, spark, dpd=DPD_THRESHOLD, mob=LABEL_MOB):
    """Create label store from loan daily data"""
    
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # load from silver loan daily
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = os.path.join(silver_directory, "silver_loan_daily", partition_name)
    df = spark.read.parquet(filepath)
    print(f'Loaded from: {filepath}, row count: {df.count()}')

    # filter for specific MOB
    df = df.filter(col("mob") == mob)

    # create label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))

    # select columns for label store
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save label store
    partition_name = "label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    output_dir = os.path.join(gold_directory, "label_store")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f'Saved to: {filepath}')
    
    return df


def process_feature_store(snapshot_date_str, silver_directory, gold_directory, spark, feature_mob=FEATURE_MOB):
    """Create unified feature store by joining all silver tables
    Feature store is created at application time (MOB=0) to prevent temporal leakage
    """
    
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # Load loan daily and filter for MOB=0 (application time only)
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = os.path.join(silver_directory, "silver_loan_daily", partition_name)
    df_loan = spark.read.parquet(filepath)
    print(f'Loaded loan_daily from: {filepath}, row count: {df_loan.count()}')
    
    # CRITICAL: Filter for MOB=feature_mob (default 0) to prevent temporal leakage
    # Only use features available at loan application time
    df_loan = df_loan.filter(col("mob") == feature_mob)
    print(f'Filtered to MOB={feature_mob} (application time): {df_loan.count()} rows')
    
    # Load attributes
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = os.path.join(silver_directory, "silver_attributes", partition_name)
    df_attributes = spark.read.parquet(filepath)
    print(f'Loaded attributes from: {filepath}, row count: {df_attributes.count()}')
    
    # Load financials
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = os.path.join(silver_directory, "silver_financials", partition_name)
    df_financials = spark.read.parquet(filepath)
    print(f'Loaded financials from: {filepath}, row count: {df_financials.count()}')
    
    # Load clickstream
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = os.path.join(silver_directory, "silver_clickstream", partition_name)
    df_clickstream = spark.read.parquet(filepath)
    print(f'Loaded clickstream from: {filepath}, row count: {df_clickstream.count()}')
    
    # Select relevant columns from loan daily
    # Exclude features that represent future loan performance
    # These features would cause temporal leakage as they're not available at MOB=0:
    # - installment_num: Payment sequence (future information)
    # - due_amt: Amount due at future installments
    # - paid_amt: Payment behavior (future)
    # - overdue_amt: Future delinquency information
    # - balance: Account balance changes over time
    # - mob: Month indicator (filtered to 0, not needed as feature)
    # - dpd: Days past due (this is what we're predicting!)
    df_loan_features = df_loan.select(
        "loan_id",
        "Customer_ID",
        "loan_start_date",
        "tenure",
        "loan_amt",
        "snapshot_date"
    )
    
    # Select relevant columns from attributes (exclude Name and SSN for privacy)
    df_attributes_features = df_attributes.select(
        "Customer_ID",
        col("Age").alias("customer_age"),
        col("Occupation").alias("customer_occupation"),
        "age_group",
        "occupation_category"
    )
    
    # Select relevant columns from financials
    df_financials_features = df_financials.select(
        "Customer_ID",
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Num_of_Loan",
        "Delay_from_due_date",
        "Num_of_Delayed_Payment",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Total_EMI_per_month",
        "Amount_invested_monthly",
        "Monthly_Balance",
        "debt_to_income_ratio",
        "income_category"
    )
    
    # Select relevant columns from clickstream
    df_clickstream_features = df_clickstream.select(
        "Customer_ID",
        "fe_1",
        "fe_2",
        "fe_3",
        "fe_4",
        "fe_5",
        "fe_6",
        "fe_7",
        "fe_8",
        "fe_9",
        "fe_10",
        "fe_11",
        "fe_12",
        "fe_13",
        "fe_14",
        "fe_15",
        "fe_16",
        "fe_17",
        "fe_18",
        "fe_19",
        "fe_20",
    )
    
    # Join all tables on Customer_ID
    df_features = df_loan_features \
        .join(df_attributes_features, on="Customer_ID", how="left") \
        .join(df_financials_features, on="Customer_ID", how="left") \
        .join(df_clickstream_features, on="Customer_ID", how="left")
    
    print(f'Feature store created with {df_features.count()} rows and {len(df_features.columns)} columns')
    
    # Save feature store
    partition_name = "feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    output_dir = os.path.join(gold_directory, "feature_store")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, partition_name)
    df_features.write.mode("overwrite").parquet(filepath)
    print(f'Saved to: {filepath}')
    
    return df_features


def process_all_gold_tables(snapshot_date_str, silver_directory, gold_directory, spark, dpd=DPD_THRESHOLD, mob=LABEL_MOB, feature_mob=FEATURE_MOB):
    """Process all gold tables: feature store and label store
    """
    
    print(f"\n{'='*70}")
    print(f"Processing Gold Layer for {snapshot_date_str}")
    print(f"Feature Store: MOB={feature_mob} (Application Time)")
    print(f"Label Store: MOB={mob}, DPD>={dpd} (6-month Default Observation)")
    print(f"{'='*70}\n")
    
    print("1. Creating Feature Store (MOB=0 - Application Time Features)...")
    df_features = process_feature_store(snapshot_date_str, silver_directory, gold_directory, spark, feature_mob=feature_mob)
    print("Feature Store completed\n")
    
    print(f"2. Creating Label Store (MOB={mob} - 6-month Default Labels)...")
    df_labels = process_label_store(snapshot_date_str, silver_directory, gold_directory, spark, dpd=dpd, mob=mob)
    print("Label Store completed\n")

    print(f"{'='*70}")
    print(f"Gold Layer Processing Complete for {snapshot_date_str}")
    print(f"Feature Store: {df_features.count()} rows, {len(df_features.columns)} columns")
    print(f"Label Store: {df_labels.count()} rows, {len(df_labels.columns)} columns")
    print(f"{'='*70}\n")
    
    return {
        'feature_store': df_features,
        'label_store': df_labels
    }