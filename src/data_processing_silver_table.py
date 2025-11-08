import os
from datetime import datetime
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_loan_daily_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    """Process loan daily data from bronze to silver layer"""
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = os.path.join(bronze_directory, "bronze_loan_daily", partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f'Loaded from: {filepath}, row count: {df.count()}')

    # clean data: enforce schema / data type
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", 
                      F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", 
                      F.when(col("installments_missed") > 0, 
                            F.add_months(col("snapshot_date"), -1 * col("installments_missed")))
                      .cast(DateType()))
    df = df.withColumn("dpd", 
                      F.when(col("overdue_amt") > 0.0, 
                            F.datediff(col("snapshot_date"), col("first_missed_date")))
                      .otherwise(0).cast(IntegerType()))

    # save silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    output_dir = os.path.join(silver_directory, "silver_loan_daily")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f'Saved to: {filepath}')
    
    return df


def process_silver_attributes_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    """Process customer attributes data from bronze to silver layer"""
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = os.path.join(bronze_directory, "bronze_attributes", partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f'Loaded from: {filepath}, row count: {df.count()}')

    # clean data: strip underscores from string columns before type conversion
    columns_to_clean = ['Age']
    for column in columns_to_clean:
        if column in df.columns:
            df = df.withColumn(column, F.regexp_replace(col(column).cast(StringType()), "_", ""))
    
    # clean data: enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "Name": StringType(),
        "Age": IntegerType(),
        "SSN": StringType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))
    
    # data validation: set invalid age values to null
    df = df.withColumn("Age", 
                      F.when((col("Age") > 150) | (col("Age") < 0), None)
                       .otherwise(col("Age")))

    # augment data: add age group category
    df = df.withColumn("age_group", 
                      F.when(col("Age") < 25, "18-24")
                       .when(col("Age") < 35, "25-34")
                       .when(col("Age") < 45, "35-44")
                       .when(col("Age") < 55, "45-54")
                       .when(col("Age") < 65, "55-64")
                       .otherwise("65+"))
    
    # augment data: add occupation category
    df = df.withColumn("occupation_category",
                      F.when(col("Occupation").isin(["Engineer", "Developer", "Scientist", "Architect"]), "Technical")
                       .when(col("Occupation").isin(["Doctor", "Lawyer", "Accountant"]), "Professional")
                       .when(col("Occupation").isin(["Manager", "Director", "Executive"]), "Management")
                       .when(col("Occupation").isin(["Teacher", "Professor"]), "Education")
                       .otherwise("Other"))

    # save silver table
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    output_dir = os.path.join(silver_directory, "silver_attributes")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f'Saved to: {filepath}')
    
    return df


def process_silver_financials_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    """Process financial data from bronze to silver layer"""
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = os.path.join(bronze_directory, "bronze_financials", partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f'Loaded from: {filepath}, row count: {df.count()}')

    # clean data: strip underscores from numeric columns before type conversion
    columns_to_clean = [
        'Annual_Income', 'Num_of_Loan', 'Num_of_Delayed_Payment', 
        'Changed_Credit_Limit', 'Outstanding_Debt', 'Amount_invested_monthly', 
        'Monthly_Balance'
    ]
    for column in columns_to_clean:
        if column in df.columns:
            df = df.withColumn(column, F.regexp_replace(col(column).cast(StringType()), "_", ""))
    
    # clean data: enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": FloatType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))
    
    # data validation: set extreme/invalid Monthly_Balance to null
    # Values like -333333333333333333333333333 are data quality placeholders
    df = df.withColumn("Monthly_Balance", 
                      F.when((col("Monthly_Balance") < -1000000) | (col("Monthly_Balance") > 1e15), None)
                       .otherwise(col("Monthly_Balance")))

    # augment data: add debt-to-income ratio
    df = df.withColumn("debt_to_income_ratio", 
                      F.when(col("Monthly_Inhand_Salary") > 0, 
                            (col("Total_EMI_per_month") / col("Monthly_Inhand_Salary")) * 100)
                       .otherwise(0).cast(FloatType()))
    
    # augment data: add income category
    df = df.withColumn("income_category",
                      F.when(col("Annual_Income") < 30000, "Low")
                       .when(col("Annual_Income") < 70000, "Medium")
                       .when(col("Annual_Income") < 120000, "High")
                       .otherwise("Very High"))
    

    # save silver table
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    output_dir = os.path.join(silver_directory, "silver_financials")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f'Saved to: {filepath}')
    
    return df


def process_silver_clickstream_table(snapshot_date_str, bronze_directory, silver_directory, spark):
    """Process clickstream data from bronze to silver layer"""
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = os.path.join(bronze_directory, "bronze_clickstream", partition_name)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f'Loaded from: {filepath}, row count: {df.count()}')

    # clean data: enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }
    
    # Add all feature columns as FloatType
    for i in range(1, 21):
        column_type_map[f"fe_{i}"] = FloatType()

    for column, new_type in column_type_map.items():
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(new_type))

    # save silver table
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    output_dir = os.path.join(silver_directory, "silver_clickstream")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f'Saved to: {filepath}')
    
    return df


def process_all_silver_tables(snapshot_date_str, bronze_directory, silver_directory, spark):
    """Process all bronze tables to silver layer"""
    
    print(f"\n{'='*70}")
    print(f"Processing Silver Tables for {snapshot_date_str}")
    print(f"{'='*70}\n")
    
    print("1. Processing Loan Daily Table...")
    df_loan_daily = process_silver_loan_daily_table(
        snapshot_date_str, bronze_directory, silver_directory, spark
    )
    print("Loan Daily completed\n")
    
    print("2. Processing Attributes Table...")
    df_attributes = process_silver_attributes_table(
        snapshot_date_str, bronze_directory, silver_directory, spark
    )
    print("Attributes completed\n")
    
    print("3. Processing Financials Table...")
    df_financials = process_silver_financials_table(
        snapshot_date_str, bronze_directory, silver_directory, spark
    )
    print("Financials completed\n")
    
    print("4. Processing Clickstream Table...")
    df_clickstream = process_silver_clickstream_table(
        snapshot_date_str, bronze_directory, silver_directory, spark
    )
    print("Clickstream completed\n")
    
    print(f"{'='*70}")
    print(f"Silver Tables Processing Complete for {snapshot_date_str}")
    print(f"{'='*70}\n")
    
    return {
        'loan_daily': df_loan_daily,
        'attributes': df_attributes,
        'financials': df_financials,
        'clickstream': df_clickstream
    }