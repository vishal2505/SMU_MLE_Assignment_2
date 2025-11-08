"""
Model Training Script
Trains Logistic Regression, Random Forest, and XGBoost models
Saves best model and evaluation metrics to model_store
"""

import os
import sys
import json
import pickle
import argparse
import warnings
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, log_loss, confusion_matrix,
    roc_curve, classification_report
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from pipeline_config.config import (
    GOLD_PATH, MODEL_STORE_PATH, FEATURE_COLUMNS,
    MODELS, RANDOM_STATE, TEMPORAL_SPLITS,
    MODEL_SELECTION_METRIC, MODEL_CONFIG_FILE, MODEL_EVALUATION_FILE,
    LABEL_MOB
)

warnings.filterwarnings('ignore')


def load_training_data(start_date, end_date, label_offset_months=0):
    """
    Load and merge feature store and label store for training
    
    Args:
        start_date: Start date for training window (YYYY-MM-DD)
        end_date: End date for training window (YYYY-MM-DD)
    
    Returns:
        DataFrame with features and labels
    """
    print(f"\n{'='*70}")
    print(f"Loading training data from {start_date} to {end_date}")
    print(f"{'='*70}\n")
    
    feature_dfs = []
    label_dfs = []
    
    # Load all feature store files in date range
    feature_dir = GOLD_PATH / "feature_store"
    for file in sorted(feature_dir.glob("feature_store_*.parquet")):
        # Extract date from filename
        date_str = file.stem.replace("feature_store_", "").replace("_", "-")
        if start_date <= date_str <= end_date:
            df = pd.read_parquet(file)
            print(f"Loaded features: {file.name} - {len(df)} rows")
            feature_dfs.append(df)
    
    # Load label store files with temporal alignment (labels observed LABEL_MOB months later)
    # If label_offset_months > 0, shift feature window forward by offset when selecting labels.
    label_dir = GOLD_PATH / "label_store"
    from dateutil.relativedelta import relativedelta as _relativedelta
    import datetime as _pydt
    start_dt = _pydt.datetime.fromisoformat(start_date)
    end_dt = _pydt.datetime.fromisoformat(end_date)
    label_start_dt = start_dt + _relativedelta(months=label_offset_months)
    label_end_dt = end_dt + _relativedelta(months=label_offset_months)
    print(f"Label alignment: offset={label_offset_months} months ⇒ selecting labels from {label_start_dt.date()} to {label_end_dt.date()}")
    for file in sorted(label_dir.glob("label_store_*.parquet")):
        date_str = file.stem.replace("label_store_", "").replace("_", "-")
        try:
            file_dt = _pydt.datetime.fromisoformat(date_str)
        except ValueError:
            continue
        if label_start_dt <= file_dt <= label_end_dt:
            df = pd.read_parquet(file)
            print(f"Loaded labels: {file.name} - {len(df)} rows")
            label_dfs.append(df)
    
    if not feature_dfs:
        raise ValueError("No feature data found in the specified date range")
    if not label_dfs:
        raise ValueError(
            f"No label data found for feature window {start_date} to {end_date} with offset {label_offset_months} months. "
            f"Expected label files covering {label_start_dt.date()} to {label_end_dt.date()}."
        )
    labels_df = pd.concat(label_dfs, ignore_index=True)
    
    # Combine all dataframes
    features_df = pd.concat(feature_dfs, ignore_index=True)
    
    print(f"\nTotal features loaded: {len(features_df)} rows")
    print(f"Total labels loaded: {len(labels_df)} rows")
    
    # Merge features and labels on loan_id
    if 'label' in labels_df.columns and len(labels_df) > 0:
        merged_df = features_df.merge(
            labels_df[['loan_id', 'label']],
            on='loan_id',
            how='inner'
        )
    else:
        merged_df = features_df.copy()
        merged_df['label'] = np.nan
    
    print(f"Merged dataset: {len(merged_df)} rows")
    label_counts = merged_df['label'].dropna().value_counts().to_dict()
    default_rate = merged_df['label'].dropna().mean() if merged_df['label'].notna().any() else float('nan')
    print(f"Label distribution: {label_counts}")
    if np.isnan(default_rate):
        print("Default rate: N/A (no labels available)\n")
    else:
        print(f"Default rate: {default_rate:.2%}\n")
    
    return merged_df


def prepare_features(df, feature_cols):
    """
    Prepare feature matrix and target vector
    
    Args:
        df: DataFrame with features and labels
        feature_cols: List of feature column names
    
    Returns:
        X: Feature matrix
        y: Target vector
        available_features: List of actually available features
    """
    # Filter to only available features
    available_features = [col for col in feature_cols if col in df.columns]
    missing_features = [col for col in feature_cols if col not in df.columns]
    
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
    
    print(f"Using {len(available_features)} features for training")
    
    # Extract features and target
    X = df[available_features].copy()
    y = df['label'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    return X, y, available_features


def train_model(model_name, model_params, X_train, y_train, X_val, y_val):
    """
    Train a single model and evaluate on validation set
    
    Args:
        model_name: Name of the model
        model_params: Model hyperparameters
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
    
    Returns:
        Dictionary with model, scaler, and evaluation metrics
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}\n")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Initialize model
    if model_name == 'LogisticRegression':
        model = LogisticRegression(**model_params)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(**model_params)
    elif model_name == 'XGBoost':
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model_params_xgb = model_params.copy()
        model_params_xgb['scale_pos_weight'] = scale_pos_weight
        model = xgb.XGBClassifier(**model_params_xgb)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Train model
    print(f"Training {model_name}...")
    model.fit(X_train_scaled, y_train)
    print(f"{model_name} training completed")
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_train_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
    
    y_val_pred = model.predict(X_val_scaled)
    y_val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_pred_proba)
    val_metrics = calculate_metrics(y_val, y_val_pred, y_val_pred_proba)
    
    # Print results
    print(f"\n{model_name} Training Metrics:")
    print(f"  AUC-ROC: {train_metrics['auc_roc']:.4f}")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  Precision: {train_metrics['precision']:.4f}")
    print(f"  Recall: {train_metrics['recall']:.4f}")
    print(f"  F1-Score: {train_metrics['f1_score']:.4f}")
    
    print(f"\n{model_name} Validation Metrics:")
    print(f"  AUC-ROC: {val_metrics['auc_roc']:.4f}")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  Precision: {val_metrics['precision']:.4f}")
    print(f"  Recall: {val_metrics['recall']:.4f}")
    print(f"  F1-Score: {val_metrics['f1_score']:.4f}")
    
    return {
        'model_name': model_name,
        'model': model,
        'scaler': scaler,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'params': model_params
    }


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate evaluation metrics with safeguards for degenerate cases"""
    m = {}
    # Always available
    m['accuracy'] = accuracy_score(y_true, y_pred)
    m['precision'] = precision_score(y_true, y_pred, zero_division=0)
    m['recall'] = recall_score(y_true, y_pred, zero_division=0)
    m['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    # Handle potential single-class issues for AUC and log loss
    try:
        m['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
    except Exception:
        m['auc_roc'] = float('nan')
    try:
        m['log_loss'] = log_loss(y_true, y_pred_proba)
    except Exception:
        m['log_loss'] = float('nan')
    return m


def save_model_artifacts(model_result, features, model_store_path):
    """
    Save model, scaler, and metadata to model store
    
    Args:
        model_result: Dictionary with model and metrics
        features: List of feature names
        model_store_path: Path to save model artifacts
    """
    model_name = model_result['model_name']
    model_dir = model_store_path / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_file = model_dir / "model.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model_result['model'], f)
    print(f"Saved model to {model_file}")
    
    # Save scaler
    scaler_file = model_dir / "scaler.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(model_result['scaler'], f)
    print(f"Saved scaler to {scaler_file}")
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'features': features,
        'train_metrics': model_result['train_metrics'],
        'val_metrics': model_result['val_metrics'],
        'test_metrics': model_result.get('test_metrics', {}),
        'params': model_result['params'],
        'trained_at': datetime.now().isoformat()
    }
    
    metadata_file = model_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to {metadata_file}")


def save_best_model_config(model_results, model_store_path):
    """
    Save configuration identifying the best model
    
    Args:
        model_results: List of model result dictionaries
        model_store_path: Path to model store
    """
    # Find best model based on validation AUC-ROC
    best_model = max(model_results, key=lambda x: x['val_metrics'][MODEL_SELECTION_METRIC])
    
    config = {
        'best_model': best_model['model_name'],
        'selection_metric': MODEL_SELECTION_METRIC,
        'selection_score': best_model['val_metrics'][MODEL_SELECTION_METRIC],
        'all_models': {
            result['model_name']: {
                'train_metrics': result['train_metrics'],
                'val_metrics': result['val_metrics'],
                'test_metrics': result.get('test_metrics', {})
            }
            for result in model_results
        },
        'updated_at': datetime.now().isoformat()
    }
    
    config_file = model_store_path / MODEL_CONFIG_FILE
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n{'='*70}")
    print(f"Best Model: {best_model['model_name']}")
    print(f"Validation {MODEL_SELECTION_METRIC}: {best_model['val_metrics'][MODEL_SELECTION_METRIC]:.4f}")
    print(f"{'='*70}\n")
    print(f"Saved best model config to {config_file}")


def save_model_evaluation(model_results, model_store_path):
    """
    Save detailed evaluation summary for all models
    
    Args:
        model_results: List of model result dictionaries
        model_store_path: Path to model store
    """
    evaluation = {
        'evaluation_date': datetime.now().isoformat(),
        'models': {}
    }
    
    for result in model_results:
        model_name = result['model_name']
        evaluation['models'][model_name] = {
            'train_metrics': result['train_metrics'],
            'val_metrics': result['val_metrics'],
            'test_metrics': result.get('test_metrics', {}),
            'params': result['params']
        }
    
    eval_file = model_store_path / MODEL_EVALUATION_FILE
    with open(eval_file, 'w') as f:
        json.dump(evaluation, f, indent=4)
    
    print(f"Saved model evaluation to {eval_file}")


def main():
    """Main training pipeline using temporal windows"""
    parser = argparse.ArgumentParser(description='Train ML models for loan default prediction')
    parser.add_argument('--use_temporal_splits', action='store_true', 
                       help='Use predefined temporal splits from config')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"ML Model Training Pipeline")
    print(f"{'='*70}\n")
    
    if args.use_temporal_splits:
        print("Using temporal splits from configuration:")
        print(f"  Train: {TEMPORAL_SPLITS['train']['start_date']} to {TEMPORAL_SPLITS['train']['end_date']}")
        print(f"  Validation: {TEMPORAL_SPLITS['validation']['start_date']} to {TEMPORAL_SPLITS['validation']['end_date']}")
        print(f"  Test: {TEMPORAL_SPLITS['test']['start_date']} to {TEMPORAL_SPLITS['test']['end_date']}")
    
    print(f"Models: {list(MODELS.keys())}")
    print(f"Selection Metric: {MODEL_SELECTION_METRIC}")
    print(f"Random State: {RANDOM_STATE}\n")
    
    # Load data for each temporal split
    print("Loading training data...")
    train_df = load_training_data(
        TEMPORAL_SPLITS['train']['start_date'],
        TEMPORAL_SPLITS['train']['end_date'],
        label_offset_months=LABEL_MOB
    )
    
    print("Loading validation data...")
    val_df = load_training_data(
        TEMPORAL_SPLITS['validation']['start_date'],
        TEMPORAL_SPLITS['validation']['end_date'],
        label_offset_months=LABEL_MOB
    )
    
    print("Loading test data...")
    test_df = load_training_data(
        TEMPORAL_SPLITS['test']['start_date'],
        TEMPORAL_SPLITS['test']['end_date'],
        label_offset_months=LABEL_MOB
    )
    
    # Prepare features for each split
    X_train, y_train, features = prepare_features(train_df, FEATURE_COLUMNS)
    X_val, y_val, _ = prepare_features(val_df, FEATURE_COLUMNS)
    X_test, y_test, _ = prepare_features(test_df, FEATURE_COLUMNS)

    # Strict enforcement: both validation and test must contain labels
    if y_val.dropna().empty:
        raise ValueError(
            "Validation set has no labels after temporal alignment. Adjust TEMPORAL_SPLITS or ensure label store generation for the aligned months."
        )
    if y_test.dropna().empty:
        raise ValueError(
            "Test set has no labels after temporal alignment. Adjust TEMPORAL_SPLITS or ensure label store generation for the aligned months."
        )
    
    print(f"\nData Split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Train all models
    model_results = []
    for model_name, model_params in MODELS.items():
        result = train_model(
            model_name, model_params,
            X_train, y_train,
            X_val, y_val
        )
        # Compute test metrics using the trained scaler and model
        X_test_scaled = result['scaler'].transform(X_test)
        y_test_pred = result['model'].predict(X_test_scaled)
        y_test_pred_proba = result['model'].predict_proba(X_test_scaled)[:, 1]
        test_metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)
        result['test_metrics'] = test_metrics
        model_results.append(result)
        save_model_artifacts(result, features, MODEL_STORE_PATH)
    
    # Save best model configuration
    save_best_model_config(model_results, MODEL_STORE_PATH)
    
    # Save evaluation summary
    save_model_evaluation(model_results, MODEL_STORE_PATH)
    
    print(f"\n{'='*70}")
    print(f"Training Pipeline Completed Successfully")
    print(f"Model artifacts saved to: {MODEL_STORE_PATH}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
