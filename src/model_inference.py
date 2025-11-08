"""
Model Inference Script
Loads best model from model_store and makes predictions on feature store data
"""

import os
import sys
import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Iterable

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from pipeline_config.config import (
    GOLD_PATH, MODEL_STORE_PATH, MODEL_CONFIG_FILE
)


def load_best_model():
    """
    Load the best model based on model_config.json
    
    Returns:
        Dictionary with model, scaler, and metadata
    """
    # Load model config
    config_file = MODEL_STORE_PATH / MODEL_CONFIG_FILE
    if not config_file.exists():
        raise FileNotFoundError(f"Model config not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    best_model_name = config['best_model']
    print(f"Loading best model: {best_model_name}")
    print(f"Selection metric: {config['selection_metric']} = {config['selection_score']:.4f}\n")
    
    # Load model artifacts
    model_dir = MODEL_STORE_PATH / best_model_name
    
    model_file = model_dir / "model.pkl"
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    scaler_file = model_dir / "scaler.pkl"
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    
    metadata_file = model_dir / "metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"Model loaded from: {model_dir}")
    print(f"Model trained at: {metadata['trained_at']}")
    print(f"Number of features: {len(metadata['features'])}\n")
    
    return {
        'model': model,
        'scaler': scaler,
        'metadata': metadata,
        'model_name': best_model_name
    }


def load_feature_store(snapshot_date: str) -> pd.DataFrame:
    """Load MOB=0 feature snapshot for the given date."""
    date_tag = datetime.strptime(snapshot_date, "%Y-%m-%d").strftime("%Y_%m_%d")
    feature_path = GOLD_PATH / "feature_store" / f"feature_store_{date_tag}.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature snapshot not found: {feature_path}")
    df = pd.read_parquet(feature_path)
    print(f"Loaded feature snapshot: {feature_path.name} | rows={len(df)}")
    return df


def month_range(start_date: str, end_date: str) -> Iterable[str]:
    """Yield YYYY-MM-DD for each month start in inclusive [start_date, end_date]."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    cur = start.replace(day=1)
    end_marker = end.replace(day=1)
    while cur <= end_marker:
        yield cur.strftime("%Y-%m-%d")
        # increment month
        year = cur.year + (cur.month // 12)
        month = 1 if cur.month == 12 else cur.month + 1
        cur = cur.replace(year=year, month=month)


def prepare_inference_features(df: pd.DataFrame, feature_names: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Select and sanitize features in training order, adding missing as NaN then median filling."""
    if 'loan_id' not in df.columns:
        raise KeyError("Column 'loan_id' missing from feature snapshot")
    if 'Customer_ID' not in df.columns:
        raise KeyError("Column 'Customer_ID' missing from feature snapshot")
    loan_ids = df['loan_id'].values
    customer_ids = df['Customer_ID'].values
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        print(f"Warning: {len(missing)} missing features → filling with NaN then median. First few: {missing[:5]}")
        for m in missing:
            df[m] = np.nan
    X = df[feature_names].copy()
    # Coerce numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    # Replace inf
    X = X.replace([np.inf, -np.inf], np.nan)
    # Median fill twice (after coercion & inf handling)
    X = X.fillna(X.median())
    print(f"Prepared features: shape={X.shape}")
    return X, loan_ids, customer_ids


def make_predictions(model_artifacts: dict, X: pd.DataFrame) -> np.ndarray:
    """Scale and score; return probability of class 1."""
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    X_scaled = scaler.transform(X.values)
    proba = model.predict_proba(X_scaled)[:, 1]
    return proba


def save_predictions(predictions: np.ndarray, loan_ids: np.ndarray, customer_ids: np.ndarray,
                     snapshot_date: str, model_name: str, threshold: float = 0.5) -> Path:
    """Persist prediction probabilities & hard labels to gold/predictions."""
    pred_labels = (predictions >= threshold).astype(int)
    out_dir = GOLD_PATH / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = snapshot_date.replace('-', '_')
    out_path = out_dir / f"predictions_{model_name}_{tag}.parquet"
    df_out = pd.DataFrame({
        'loan_id': loan_ids,
        'Customer_ID': customer_ids,
        'prediction_proba': predictions,
        'prediction_label': pred_labels,
        'threshold': threshold,
        'model_name': model_name,
        'inference_date': snapshot_date,
        'inference_timestamp': datetime.now().isoformat()
    })
    df_out.to_parquet(out_path, index=False)
    print(f"Saved predictions: {out_path} | rows={len(df_out)} | mean_proba={predictions.mean():.4f} | default_rate={pred_labels.mean():.2%}")
    return out_path


def main():
    """Main inference pipeline"""
    parser = argparse.ArgumentParser(description='Run model inference for loan default prediction')
    parser.add_argument('--snapshot_date', type=str, help='Single snapshot date (YYYY-MM-DD)')
    parser.add_argument('--start_date', type=str, help='Start date inclusive (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date inclusive (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"ML Model Inference Pipeline")
    print(f"{'='*70}\n")
    # Determine mode: single date or range
    if not args.snapshot_date and not (args.start_date and args.end_date):
        raise SystemExit("Provide either --snapshot_date or both --start_date and --end_date")

    # Load best model once
    model_artifacts = load_best_model()

    dates = [args.snapshot_date] if args.snapshot_date else list(month_range(args.start_date, args.end_date))
    print(f"Running inference for {len(dates)} month(s): {dates[0]} ... {dates[-1]}\n")

    successes = 0
    for dt in dates:
        print(f"--- Inference for {dt} ---")
        try:
            features_df = load_feature_store(dt)
        except FileNotFoundError as e:
            print(f"[Skip] {e}")
            continue
        # Prepare features
        X, loan_ids, customer_ids = prepare_inference_features(features_df, model_artifacts['metadata']['features'])
        # Score
        print("Scoring...")
        predictions = make_predictions(model_artifacts, X)
        # Persist
        save_predictions(predictions, loan_ids, customer_ids, dt, model_artifacts['model_name'])
        successes += 1

    print(f"Inference pipeline completed. Successful months: {successes}/{len(dates)}\n")


if __name__ == "__main__":
    main()
