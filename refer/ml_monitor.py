import argparse
import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# Model 1 Monitoring Script
# Purpose: Calculate performance metrics for Model 1 predictions
# Input: Predictions + labels (ground truth)
# Output: datamart/gold/monitoring/model_1_metrics_YYYY_MM_DD.parquet

DATAMART_ROOT_ENV_VAR = "DATAMART_ROOT"
DATAMART_ROOT_CANDIDATES = [
    os.environ.get(DATAMART_ROOT_ENV_VAR),
    "datamart",
    "scripts/datamart",
    "/opt/airflow/scripts/datamart",
]


def get_datamart_roots():
    """Return ordered list of candidate datamart roots (deduplicated)."""
    roots = []
    for candidate in DATAMART_ROOT_CANDIDATES:
        if candidate and candidate not in roots:
            roots.append(candidate.rstrip("/"))
    if not roots:
        roots.append("datamart")
    return roots


def resolve_datamart_path(relative_path, expect_directory=False):
    """Find the first matching datamart path for given relative path."""
    attempted = []
    for root in get_datamart_roots():
        candidate = os.path.join(root, relative_path)
        attempted.append(candidate)
        if expect_directory and os.path.isdir(candidate):
            return candidate, root, attempted
        if not expect_directory and os.path.exists(candidate):
            return candidate, root, attempted
    return None, None, attempted


def extract_loan_start_dates(df, id_column="loan_id"):
    """Extract distinct loan start dates encoded in loan_id suffix."""
    pattern = r"_(\d{4}_\d{2}_\d{2})$"
    start_dates = (
        df.select(F.regexp_extract(col(id_column), pattern, 1).alias("loan_start_date"))
        .filter(col("loan_start_date") != "")
        .distinct()
        .collect()
    )
    return sorted({row.loan_start_date for row in start_dates if row.loan_start_date})


def infer_prediction_dates_from_labels(df_labels):
    """Infer candidate prediction snapshot dates from label loan_ids."""
    print("\nInspecting labels to infer prediction cohorts...")
    label_start_dates = extract_loan_start_dates(df_labels)
    if label_start_dates:
        print(f"  Loan start dates present in labels: {label_start_dates}")
    else:
        print(
            "  Unable to infer loan start dates from labels; will rely on 6-month offset."
        )
    return label_start_dates


def load_predictions(snapshot_date_str, spark, expected_prediction_dates=None):
    """Load predictions from gold predictions table

    TEMPORAL CORRECTION:
    - Labels are created for loans at mob=6 on the snapshot_date
    - These loans were at mob=0 exactly 6 months earlier
    - So we need predictions from 6 months before the snapshot_date
    """
    from dateutil.relativedelta import relativedelta

    predictions_dir = "gold/predictions/"
    snapshot_date_obj = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    candidate_dates = []

    if expected_prediction_dates:
        for date_str in expected_prediction_dates:
            try:
                candidate = datetime.strptime(date_str, "%Y_%m_%d").strftime("%Y-%m-%d")
                if candidate not in candidate_dates:
                    candidate_dates.append(candidate)
            except ValueError:
                continue

    # Always prioritise the 6-month offset from the monitoring snapshot
    six_month_offset = (snapshot_date_obj - relativedelta(months=6)).strftime(
        "%Y-%m-%d"
    )
    if six_month_offset not in candidate_dates:
        candidate_dates.insert(0, six_month_offset)

    # Add same-day fallback last in case the historical file is missing
    snapshot_date_formatted = snapshot_date_obj.strftime("%Y-%m-%d")
    if snapshot_date_formatted not in candidate_dates:
        candidate_dates.append(snapshot_date_formatted)

    print("\nSearching for predictions to pair with labels...")
    all_attempts = []
    for candidate in candidate_dates:
        file_date_str = datetime.strptime(candidate, "%Y-%m-%d").strftime("%Y_%m_%d")
        relative_path = os.path.join(
            predictions_dir, f"model_1_predictions_{file_date_str}.parquet"
        )
        predictions_file, datamart_root, attempts = resolve_datamart_path(relative_path)
        all_attempts.extend(attempts)
        print(f"  → Checking {relative_path}")
        if not predictions_file:
            print("     ✗ File not found in any candidate datamart root")
            for attempt in attempts:
                print(f"       - {attempt}")
            continue

        print(f"     ✓ Found predictions generated for {candidate}")
        print(f"       Resolved path: {predictions_file}")
        df_predictions = spark.read.parquet(predictions_file)

        row_count = df_predictions.count()
        print(f"\nLoading predictions from: {predictions_file}")
        print(f"  Prediction date (mob=0): {candidate}")
        print(f"  Label date (mob=6):      {snapshot_date_str}")
        print("  These loans should match after 6-month maturation period")
        print(f"Loaded predictions: {row_count} rows")
        print(f"Columns: {df_predictions.columns}")

        loan_start_dates = extract_loan_start_dates(df_predictions)
        if loan_start_dates:
            print(f"  Loan start dates present in predictions: {loan_start_dates}")
        else:
            print(
                "  Warning: could not infer loan start dates from prediction loan_ids"
            )

        if "inference_date" in df_predictions.columns:
            inference_dates = [
                row.inference_date
                for row in df_predictions.select("inference_date").distinct().collect()
            ]
            print(f"  Inference dates recorded in file: {inference_dates}")

        return df_predictions, candidate, predictions_file, datamart_root

    print("⚠️  Predictions file not found for any of the candidate dates:")
    for path in all_attempts:
        print(f"   - {path}")
    raise FileNotFoundError(
        f"Could not locate predictions for snapshot {snapshot_date_str}. "
        f"Tried: {', '.join(candidate_dates)}"
    )


def load_labels(snapshot_date_str, spark):
    """Load ground truth labels from gold label store"""

    label_store_dir = "gold/label_store/"

    # Build path for the specific snapshot date
    date_obj = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    file_date_str = date_obj.strftime("%Y_%m_%d")
    relative_path = os.path.join(
        label_store_dir, f"gold_label_store_{file_date_str}.parquet"
    )
    label_file, datamart_root, attempts = resolve_datamart_path(relative_path)

    if not label_file:
        print("⚠️  Label file not found in any candidate datamart root:")
        for path in attempts:
            print(f"   - {path}")
        raise FileNotFoundError(f"Label file not found: {relative_path}")

    print(f"\nLoading labels from: {label_file}")
    df_labels = spark.read.parquet(label_file)

    print(f"Loaded labels: {df_labels.count()} rows")
    print(f"Columns: {df_labels.columns}")

    return df_labels, label_file, datamart_root


def join_predictions_and_labels(df_predictions, df_labels):
    """Join predictions with ground truth labels"""

    print("\n" + "=" * 60)
    print("Joining Predictions with Labels")
    print("=" * 60)

    # Join on loan_id and Customer_ID
    df_joined = df_predictions.join(
        df_labels.select("loan_id", "Customer_ID", "label", "label_def"),
        on=["loan_id", "Customer_ID"],
        how="inner",
    )

    joined_count = df_joined.count()
    print(f"Joined data: {joined_count} rows")

    if joined_count == 0:
        pred_starts = extract_loan_start_dates(df_predictions)
        label_starts = extract_loan_start_dates(df_labels)
        sample_pred_ids = [
            row.loan_id for row in df_predictions.select("loan_id").limit(5).collect()
        ]
        sample_label_ids = [
            row.loan_id for row in df_labels.select("loan_id").limit(5).collect()
        ]
        diagnostic_message = (
            "No matching records found between predictions and labels.\n"
            f"  Prediction loan start dates: {pred_starts}\n"
            f"  Label loan start dates: {label_starts}\n"
            f"  Sample prediction loan_ids: {sample_pred_ids}\n"
            f"  Sample label loan_ids: {sample_label_ids}"
        )
        raise ValueError(diagnostic_message)

    return df_joined


def calculate_metrics(df_joined):
    """Calculate performance metrics"""

    print("\n" + "=" * 60)
    print("Calculating Performance Metrics")
    print("=" * 60)

    # Convert to pandas for sklearn metrics
    pdf = df_joined.select(
        "loan_id", "Customer_ID", "prediction_proba", "prediction_label", "label"
    ).toPandas()

    y_true = pdf["label"].values
    y_pred = pdf["prediction_label"].values
    y_proba = pdf["prediction_proba"].values

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Handle case where only one class is present
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError as e:
        print(f"⚠️  Warning: Could not calculate ROC-AUC: {e}")
        roc_auc = None

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Calculate additional metrics
    total_samples = len(y_true)
    actual_positives = y_true.sum()
    actual_negatives = total_samples - actual_positives
    predicted_positives = y_pred.sum()
    predicted_negatives = total_samples - predicted_positives

    # Prediction distribution statistics
    mean_proba = y_proba.mean()
    std_proba = y_proba.std()
    min_proba = y_proba.min()
    max_proba = y_proba.max()
    median_proba = np.median(y_proba)

    # Print summary
    print("\nPerformance Metrics Summary:")
    print(f"{'─' * 60}")
    print("Overall Metrics:")
    print(f"  Accuracy:   {accuracy:.4f}")
    if roc_auc is not None:
        print(f"  ROC-AUC:    {roc_auc:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1-Score:   {f1:.4f}")

    print("\nConfusion Matrix:")
    print("                Predicted")
    print("               No Default  Default")
    print(f"Actual No Def    {tn:6d}     {fp:6d}")
    print(f"Actual Default   {fn:6d}     {tp:6d}")

    print("\nClass Distribution:")
    print(
        f"  Actual Positives (Default):     {actual_positives:6d} ({100 * actual_positives / total_samples:5.2f}%)"
    )
    print(
        f"  Actual Negatives (No Default):  {actual_negatives:6d} ({100 * actual_negatives / total_samples:5.2f}%)"
    )
    print(
        f"  Predicted Positives (Default):  {predicted_positives:6d} ({100 * predicted_positives / total_samples:5.2f}%)"
    )
    print(
        f"  Predicted Negatives (No Default): {predicted_negatives:6d} ({100 * predicted_negatives / total_samples:5.2f}%)"
    )

    print("\nPrediction Probability Statistics:")
    print(f"  Mean:   {mean_proba:.4f}")
    print(f"  Median: {median_proba:.4f}")
    print(f"  Std:    {std_proba:.4f}")
    print(f"  Min:    {min_proba:.4f}")
    print(f"  Max:    {max_proba:.4f}")

    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["No Default", "Default"]))

    # Compile metrics into dictionary
    metrics = {
        "model_id": "model_1",
        "model_type": "logistic_regression",
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "total_samples": int(total_samples),
        "actual_positives": int(actual_positives),
        "actual_negatives": int(actual_negatives),
        "predicted_positives": int(predicted_positives),
        "predicted_negatives": int(predicted_negatives),
        "mean_prediction_proba": float(mean_proba),
        "median_prediction_proba": float(median_proba),
        "std_prediction_proba": float(std_proba),
        "min_prediction_proba": float(min_proba),
        "max_prediction_proba": float(max_proba),
    }

    return metrics


def save_metrics(metrics, snapshot_date_str, spark, datamart_root=None):
    """Save monitoring metrics to gold monitoring table"""

    monitoring_subdir = "gold/monitoring/"
    base_root = (
        datamart_root
        or resolve_datamart_path(monitoring_subdir, expect_directory=True)[1]
    )
    if base_root is None:
        base_root = get_datamart_roots()[0]
    monitoring_dir = os.path.join(base_root, monitoring_subdir)

    if not os.path.exists(monitoring_dir):
        os.makedirs(monitoring_dir)
        print(f"✓ Created monitoring directory: {monitoring_dir}")

    # Build output path
    date_obj = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    file_date_str = date_obj.strftime("%Y_%m_%d")
    output_file = f"{monitoring_dir}model_1_metrics_{file_date_str}.parquet"

    # Add metadata
    metrics["snapshot_date"] = snapshot_date_str
    metrics["monitoring_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert to DataFrame and save
    metrics_pdf = pd.DataFrame([metrics])
    df_metrics = spark.createDataFrame(metrics_pdf)

    df_metrics.write.mode("overwrite").parquet(output_file)

    print(f"\n✅ Saved metrics to: {output_file}")
    print(f"   Columns: {df_metrics.columns}")
    print(f"   Rows: {df_metrics.count()}")

    # Also save as JSON for easy reading
    json_output_file = output_file.replace(".parquet", ".json")
    with open(json_output_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved metrics (JSON) to: {json_output_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Model 1 Monitoring: Calculate performance metrics"
    )
    parser.add_argument(
        "--snapshotdate",
        type=str,
        required=True,
        help="Snapshot date for monitoring (YYYY-MM-DD)",
    )

    args = parser.parse_args()
    snapshotdate = args.snapshotdate

    print("=" * 60)
    print("Model 1 Monitoring - Logistic Regression")
    print("=" * 60)
    print(f"Snapshot Date: {snapshotdate}")
    print("=" * 60 + "\n")

    # Initialize Spark
    spark = (
        pyspark.sql.SparkSession.builder.appName("Model1_Monitoring")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        # Step 1: Load labels
        print("\n[Step 1/5] Loading labels...")
        df_labels, label_path, label_root = load_labels(snapshotdate, spark)

        label_start_dates = infer_prediction_dates_from_labels(df_labels)

        # Step 2: Load predictions
        print("\n[Step 2/5] Loading predictions...")
        df_predictions, prediction_snapshot_date, prediction_path, prediction_root = (
            load_predictions(snapshotdate, spark, label_start_dates)
        )

        # Step 3: Join predictions and labels
        print("\n[Step 3/5] Joining predictions and labels...")
        df_joined = join_predictions_and_labels(df_predictions, df_labels)

        # Step 4: Calculate and save metrics
        print("\n[Step 4/5] Calculating metrics...")
        metrics = calculate_metrics(df_joined)
        metrics["prediction_snapshot_date"] = prediction_snapshot_date
        metrics["label_snapshot_date"] = snapshotdate
        metrics["prediction_file_path"] = prediction_path
        metrics["label_file_path"] = label_path

        # Step 5: Save metrics
        print("\n[Step 5/5] Saving metrics...")
        datamart_root = prediction_root or label_root
        output_file = save_metrics(metrics, snapshotdate, spark, datamart_root)

        print("\n" + "=" * 60)
        print("✅ Model 1 Monitoring Completed Successfully")
        print("=" * 60)
        print(f"Output: {output_file}")
        print("Key Metrics:")
        print(
            f"  Prediction snapshot: {metrics.get('prediction_snapshot_date', 'unknown')}"
        )
        print(
            f"  Label snapshot:      {metrics.get('label_snapshot_date', snapshotdate)}"
        )
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        if metrics["roc_auc"] is not None:
            print(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Error during monitoring: {str(e)}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()