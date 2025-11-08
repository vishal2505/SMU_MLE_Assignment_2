import argparse
import os
import glob
import json
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit
from pyspark.sql.types import DoubleType, IntegerType
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Model 1 Inference Script
# Purpose: Load trained Model 1 (Logistic Regression) and generate predictions
# Input: Gold feature store + trained model artifacts
# Output: datamart/gold/predictions/model_1_predictions_YYYY_MM_DD.parquet

DATAMART_ROOT_ENV_VAR = "DATAMART_ROOT"
DATAMART_ROOT_CANDIDATES = [
    os.environ.get(DATAMART_ROOT_ENV_VAR),
    "datamart",
    "scripts/datamart",
    "/opt/airflow/scripts/datamart",
]

MODEL_STORE_ROOT_CANDIDATES = [
    "model_store",
    "scripts/model_store",
    "/opt/airflow/scripts/model_store",
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


def get_model_store_roots():
    """Return ordered list of candidate model store roots (deduplicated)."""
    roots = []
    for candidate in MODEL_STORE_ROOT_CANDIDATES:
        if candidate and candidate not in roots:
            roots.append(candidate.rstrip("/"))
    if not roots:
        roots.append("model_store")
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


def resolve_model_store_path(relative_path, expect_directory=False):
    """Find the first matching model store path for given relative path."""
    attempted = []
    for root in get_model_store_roots():
        candidate = os.path.join(root, relative_path)
        attempted.append(candidate)
        if expect_directory and os.path.isdir(candidate):
            return candidate, root, attempted
        if not expect_directory and os.path.exists(candidate):
            return candidate, root, attempted
    return None, None, attempted


def load_model_artifacts(model_store_dir="model_1/"):
    """Load trained model, preprocessing pipeline, and feature list"""

    # Resolve model store directory path
    resolved_dir, root, attempted = resolve_model_store_path(model_store_dir, expect_directory=True)

    if not resolved_dir:
        print("⚠️  Model store directory not found in any candidate location:")
        for path in attempted:
            print(f"   - {path}")
        raise FileNotFoundError(f"Model store directory not found: {model_store_dir}")

    model_store_dir = resolved_dir
    print(f"✓ Found model store: {model_store_dir}")

    # Load model
    model_path = os.path.join(model_store_dir, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    print(f"✓ Loaded model from: {model_path}")

    # Load preprocessing pipeline (imputer + scaler)
    preprocessing_path = os.path.join(model_store_dir, "preprocessing.pkl")
    if not os.path.exists(preprocessing_path):
        raise FileNotFoundError(f"Preprocessing pipeline not found: {preprocessing_path}")
    preprocessing = joblib.load(preprocessing_path)
    print(f"✓ Loaded preprocessing from: {preprocessing_path}")

    # Load feature list
    feature_list_path = os.path.join(model_store_dir, "features.json")
    if not os.path.exists(feature_list_path):
        raise FileNotFoundError(f"Feature list not found: {feature_list_path}")
    with open(feature_list_path, 'r') as f:
        feature_data = json.load(f)
    feature_cols = feature_data['features']
    print(f"✓ Loaded {len(feature_cols)} features")

    # Load metadata (optional, for logging)
    metadata_path = os.path.join(model_store_dir, "metadata.json")
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Model trained on: {metadata.get('training_date', 'Unknown')}")

    return model, preprocessing, feature_cols, metadata


def load_inference_data(snapshot_date_str, spark, feature_cols):
    """Load feature data for inference from gold feature store"""

    gold_feature_dir = "gold/feature_store/"

    # Build path for the specific snapshot date
    date_obj = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    file_date_str = date_obj.strftime("%Y_%m_%d")
    relative_path = os.path.join(gold_feature_dir, f"gold_feature_store_{file_date_str}.parquet")

    feature_file, datamart_root, attempted = resolve_datamart_path(relative_path)

    if not feature_file:
        print("⚠️  Feature file not found in any candidate datamart root:")
        for path in attempted:
            print(f"   - {path}")
        raise FileNotFoundError(f"Feature file not found: {relative_path}")

    print(f"\nLoading inference data from: {feature_file}")
    df_features = spark.read.parquet(feature_file)

    print(f"Loaded features: {df_features.count()} rows")

    # Ensure required columns exist
    missing_cols = [col for col in feature_cols if col not in df_features.columns]
    if missing_cols:
        print(f"⚠️  Warning: {len(missing_cols)} features missing from data")
        print(f"    Missing features: {missing_cols[:5]}...")
        # Add missing columns with null values
        for col_name in missing_cols:
            df_features = df_features.withColumn(col_name, lit(None).cast(DoubleType()))

    return df_features


def generate_predictions(model, preprocessing, df_features, feature_cols):
    """Generate predictions using trained model"""

    print("\n" + "="*60)
    print("Generating Predictions")
    print("="*60)

    # Convert to pandas for sklearn prediction
    # Keep loan_id and Customer_ID for joining back
    id_cols = ['loan_id', 'Customer_ID']
    df_with_ids = df_features.select(id_cols + feature_cols)

    pdf = df_with_ids.toPandas()
    loan_ids = pdf[id_cols].copy()

    # Extract features in correct order
    X_inference = pdf[feature_cols].copy()

    # CRITICAL: Convert non-numeric columns to numeric (same as training)
    # This handles categorical features by converting them to NaN (to be imputed)
    for col_name in X_inference.columns:
        X_inference[col_name] = pd.to_numeric(X_inference[col_name], errors='coerce')

    X_inference = X_inference.values

    print(f"Inference data shape: {X_inference.shape}")

    # Apply preprocessing (imputer + scaler)
    imputer, scaler = preprocessing['imputer'], preprocessing['scaler']
    X_imputed = imputer.transform(X_inference)
    X_scaled = scaler.transform(X_imputed)

    # Generate predictions
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]  # Probability of class 1 (default)

    # Create results dataframe
    results_pdf = loan_ids.copy()
    results_pdf['prediction_proba'] = y_proba
    results_pdf['prediction_label'] = y_pred
    results_pdf['model_id'] = 'model_1'
    results_pdf['model_type'] = 'logistic_regression'

    # Summary statistics
    print(f"\nPrediction Summary:")
    print(f"  Total predictions: {len(results_pdf)}")
    print(f"  Predicted class 1 (default): {y_pred.sum()} ({100*y_pred.mean():.2f}%)")
    print(f"  Predicted class 0 (no default): {(1-y_pred).sum()} ({100*(1-y_pred).mean():.2f}%)")
    print(f"  Mean probability: {y_proba.mean():.4f}")
    print(f"  Std probability: {y_proba.std():.4f}")
    print(f"  Min probability: {y_proba.min():.4f}")
    print(f"  Max probability: {y_proba.max():.4f}")

    return results_pdf


def save_predictions(results_pdf, snapshot_date_str, spark):
    """Save predictions to gold predictions table"""

    predictions_subdir = "gold/predictions/"

    # Try to find existing predictions directory or use first datamart root
    base_root = resolve_datamart_path(predictions_subdir, expect_directory=True)[1]
    if base_root is None:
        base_root = get_datamart_roots()[0]

    predictions_dir = os.path.join(base_root, predictions_subdir)

    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
        print(f"✓ Created predictions directory: {predictions_dir}")

    # Build output path
    date_obj = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    file_date_str = date_obj.strftime("%Y_%m_%d")
    output_file = f"{predictions_dir}model_1_predictions_{file_date_str}.parquet"

    # Add inference timestamp
    results_pdf['inference_date'] = snapshot_date_str
    results_pdf['inference_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Convert back to Spark and save
    df_predictions = spark.createDataFrame(results_pdf)

    df_predictions.write.mode('overwrite').parquet(output_file)

    print(f"\n✅ Saved predictions to: {output_file}")
    print(f"   Columns: {df_predictions.columns}")
    print(f"   Rows: {df_predictions.count()}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Model 1 Inference: Generate predictions using trained Logistic Regression')
    parser.add_argument('--snapshotdate', type=str, required=True,
                        help='Snapshot date for inference (YYYY-MM-DD)')
    parser.add_argument('--model_store', type=str, default='model_1/',
                        help='Path to model store directory (relative to model_store root)')

    args = parser.parse_args()
    snapshotdate = args.snapshotdate
    model_store_dir = args.model_store

    print("="*60)
    print(f"Model 1 Inference - Logistic Regression")
    print("="*60)
    print(f"Snapshot Date: {snapshotdate}")
    print(f"Model Store: {model_store_dir}")
    print("="*60 + "\n")

    # Initialize Spark
    spark = pyspark.sql.SparkSession.builder \
        .appName("Model1_Inference") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    try:
        # Step 1: Load model artifacts
        print("\n[Step 1/4] Loading model artifacts...")
        model, preprocessing, feature_cols, metadata = load_model_artifacts(model_store_dir)

        # Step 2: Load inference data
        print("\n[Step 2/4] Loading inference data...")
        df_features = load_inference_data(snapshotdate, spark, feature_cols)

        # Step 3: Generate predictions
        print("\n[Step 3/4] Generating predictions...")
        results_pdf = generate_predictions(model, preprocessing, df_features, feature_cols)

        # Step 4: Save predictions
        print("\n[Step 4/4] Saving predictions...")
        output_file = save_predictions(results_pdf, snapshotdate, spark)

        print("\n" + "="*60)
        print("✅ Model 1 Inference Completed Successfully")
        print("="*60)
        print(f"Output: {output_file}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()