import argparse
import os
import glob
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

# Enhanced version with explicit temporal split configuration (supports multiple OOT periods)
# to call this script: python model_2_automl_v2.py --snapshotdate "2024-12-01" --config model_config.json


def calculate_relative_windows(snapshot_date_str, relative_config):
    """
    Calculate temporal windows dynamically based on snapshot date and relative offsets.
    Works BACKWARD from snapshot_date (latest available data).

    Logic: snapshot_date represents the latest data available.
    OOT uses the most recent data, then we work backwards: test, validation, training.

    This enables rolling window retraining in production.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # Get configuration
    train_months_back = relative_config['train']['months_back']
    val_months = relative_config['validation']['months_after_train_end']
    test_months = relative_config['test']['months_after_validation_end']
    oot_months = relative_config['oot']['months_after_test_end']

    # OOT period: uses the latest available data (ending at snapshot_date)
    oot_end = snapshot_date
    oot_start = oot_end - relativedelta(months=oot_months - 1)

    # Test period: comes before OOT
    test_end = oot_start - relativedelta(months=1)
    test_start = test_end - relativedelta(months=test_months - 1)

    # Validation period: comes before test
    val_end = test_start - relativedelta(months=1)
    val_start = val_end - relativedelta(months=val_months - 1)

    # Training period: comes before validation
    train_end = val_start - relativedelta(months=1)
    train_start = train_end - relativedelta(months=train_months_back - 1)

    return {
        'train': {
            'start_date': train_start.strftime("%Y-%m-%d"),
            'end_date': train_end.strftime("%Y-%m-%d"),
            'description': f'Training - {train_months_back} months'
        },
        'validation': {
            'start_date': val_start.strftime("%Y-%m-%d"),
            'end_date': val_end.strftime("%Y-%m-%d"),
            'description': f'Validation - {val_months} months'
        },
        'test': {
            'start_date': test_start.strftime("%Y-%m-%d"),
            'end_date': test_end.strftime("%Y-%m-%d"),
            'description': f'Test - {test_months} months'
        },
        'oot': {
            'start_date': oot_start.strftime("%Y-%m-%d"),
            'end_date': oot_end.strftime("%Y-%m-%d"),
            'description': f'OOT - {oot_months} month(s)'
        }
    }


def load_config(config_path="model_config.json", snapshot_date_str=None):
    """Load training configuration from JSON file"""
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found.")
        return None

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Check if we should use relative windows
    mode = config.get('temporal_window_mode', 'absolute')

    if mode == 'relative' and snapshot_date_str:
        print(f"\n🔄 Using RELATIVE temporal windows (dynamic/production mode)")
        print(f"   Calculating windows based on snapshot_date: {snapshot_date_str}\n")

        relative_config = config['relative_windows']
        temporal_splits = calculate_relative_windows(snapshot_date_str, relative_config)
        config['temporal_splits'] = temporal_splits
    else:
        if mode == 'relative' and not snapshot_date_str:
            print(f"\n⚠️  WARNING: Config set to 'relative' mode but no snapshot_date provided.")
            print(f"   Falling back to ABSOLUTE temporal splits from config.\n")
        else:
            print(f"\n📅 Using ABSOLUTE temporal windows (fixed dates)\n")

    temporal_splits = config['temporal_splits']

    # Detect OOT periods (oot, oot1, oot2, etc.)
    oot_keys = sorted([k for k in temporal_splits.keys() if k.startswith('oot')])

    print("="*60)
    print("Loaded Training Configuration")
    print("="*60)
    print(f"Train period:      {temporal_splits['train']['start_date']} to {temporal_splits['train']['end_date']}")
    print(f"Validation period: {temporal_splits['validation']['start_date']} to {temporal_splits['validation']['end_date']}")
    print(f"Test period:       {temporal_splits['test']['start_date']} to {temporal_splits['test']['end_date']}")

    for oot_key in oot_keys:
        oot_period = temporal_splits[oot_key]
        print(f"{oot_key.upper()} period:        {oot_period['start_date']} to {oot_period['end_date']}")

    print("="*60 + "\n")

    return config


def load_training_data(snapshot_date_str, spark, config=None):
    """Load all gold tables up to snapshot_date"""
    gold_feature_dir = "datamart/gold/feature_store/"
    gold_label_dir = "datamart/gold/label_store/"

    feature_files = sorted(glob.glob(f"{gold_feature_dir}gold_feature_store_*.parquet"))
    label_files = sorted(glob.glob(f"{gold_label_dir}gold_label_store_*.parquet"))

    if not feature_files or not label_files:
        print(f"Error: No training data found")
        return None, None

    snapshot_dt = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    def extract_date_from_filename(filepath):
        filename = os.path.basename(filepath)
        date_part = filename.replace('gold_feature_store_', '').replace('gold_label_store_', '').replace('.parquet', '')
        return datetime.strptime(date_part, "%Y_%m_%d")

    valid_feature_files = [f for f in feature_files if extract_date_from_filename(f) <= snapshot_dt]
    valid_label_files = [f for f in label_files if extract_date_from_filename(f) <= snapshot_dt]

    print(f"Found {len(valid_feature_files)} feature files and {len(valid_label_files)} label files")

    if not valid_feature_files or not valid_label_files:
        return None, None

    # Load and union
    df_features_list = [spark.read.parquet(f) for f in valid_feature_files]
    df_features = df_features_list[0]
    for df in df_features_list[1:]:
        df_features = df_features.union(df)

    df_labels_list = [spark.read.parquet(f) for f in valid_label_files]
    df_labels = df_labels_list[0]
    for df in df_labels_list[1:]:
        df_labels = df_labels.union(df)

    print(f"Loaded features: {df_features.count()} rows")
    print(f"Loaded labels: {df_labels.count()} rows")

    # Check if labels exist
    if df_labels.count() == 0:
        print(f"Error: No labels found up to {snapshot_date_str}")
        print(f"Insufficient data for model training. Need data through at least validation period.")
        return None, None

    df_train = df_features.join(
        df_labels.select("loan_id", "Customer_ID", "label", "label_def"),
        on=["loan_id", "Customer_ID"],
        how="inner"
    )

    print(f"Joined training data: {df_train.count()} rows")

    # Check if join produced any results
    if df_train.count() == 0:
        print(f"Error: No training data after joining features and labels")
        print(f"Insufficient data for model training.")
        return None, None

    return df_train, df_labels.select("label_def").first()["label_def"]


def prepare_ml_dataset_with_config(df_train, spark, config):
    """Prepare dataset with explicit temporal splits (supports multiple OOT)"""
    df_pd = df_train.toPandas()

    exclude_cols = ['loan_id', 'Customer_ID', 'label', 'label_def', 'snapshot_date',
                    'feature_snapshot_date', 'mob']
    feature_cols = [c for c in df_pd.columns if c not in exclude_cols]

    X = df_pd[feature_cols].copy()
    for col_name in X.columns:
        X[col_name] = pd.to_numeric(X[col_name], errors='coerce')

    y = df_pd['label'].values
    dates = pd.to_datetime(df_pd['snapshot_date'])

    temporal_splits = config['temporal_splits']

    train_start = pd.to_datetime(temporal_splits['train']['start_date'])
    train_end = pd.to_datetime(temporal_splits['train']['end_date'])
    val_start = pd.to_datetime(temporal_splits['validation']['start_date'])
    val_end = pd.to_datetime(temporal_splits['validation']['end_date'])
    test_start = pd.to_datetime(temporal_splits['test']['start_date'])
    test_end = pd.to_datetime(temporal_splits['test']['end_date'])

    # Detect all OOT periods
    oot_keys = sorted([k for k in temporal_splits.keys() if k.startswith('oot')])
    oot_periods = {}
    for oot_key in oot_keys:
        oot_start = pd.to_datetime(temporal_splits[oot_key]['start_date'])
        oot_end = pd.to_datetime(temporal_splits[oot_key]['end_date'])
        oot_periods[oot_key] = {'start': oot_start, 'end': oot_end}

    # Create masks
    train_mask = (dates >= train_start) & (dates <= train_end)
    val_mask = (dates >= val_start) & (dates <= val_end)
    test_mask = (dates >= test_start) & (dates <= test_end)

    oot_masks = {}
    for oot_key, period in oot_periods.items():
        oot_masks[oot_key] = (dates >= period['start']) & (dates <= period['end'])

    # Split data
    data_splits = {
        'X_train': X[train_mask],
        'y_train': y[train_mask],
        'X_val': X[val_mask],
        'y_val': y[val_mask],
        'X_test': X[test_mask],
        'y_test': y[test_mask],
        'feature_cols': feature_cols
    }

    # Add OOT splits
    for oot_key, mask in oot_masks.items():
        data_splits[f'X_{oot_key}'] = X[mask]
        data_splits[f'y_{oot_key}'] = y[mask]

    split_counts = {
        'train': len(data_splits['X_train']),
        'val': len(data_splits['X_val']),
        'test': len(data_splits['X_test'])
    }
    for oot_key in oot_keys:
        split_counts[oot_key] = len(data_splits[f'X_{oot_key}'])

    data_splits['split_counts'] = split_counts

    # Print summary
    print(f"\n{'='*60}")
    print("Temporal Split Summary")
    print(f"{'='*60}")
    print(f"Training:   {train_mask.sum():6d} samples | {train_start.date()} to {train_end.date()}")
    print(f"Validation: {val_mask.sum():6d} samples | {val_start.date()} to {val_end.date()}")
    print(f"Test:       {test_mask.sum():6d} samples | {test_start.date()} to {test_end.date()}")

    for oot_key in oot_keys:
        oot_period = oot_periods[oot_key]
        print(f"{oot_key.upper()}:        {oot_masks[oot_key].sum():6d} samples | {oot_period['start'].date()} to {oot_period['end'].date()}")

    print(f"Features:   {len(feature_cols)} columns")
    empty_splits = [name.upper() for name, count in split_counts.items() if count == 0]
    if empty_splits:
        print(f"\n⚠️  Warning: The following splits have 0 samples after temporal filtering: {', '.join(empty_splits)}")
        print("    Gradient Boosting training will skip preprocessing/evaluation for these splits.")
    print(f"{'='*60}\n")

    data_splits['oot_keys'] = oot_keys

    return data_splits


def train_and_evaluate_gradient_boosting(data_splits, config):
    """Train GBM and evaluate on all splits (train/val/test/OOT*)"""
    X_train = data_splits['X_train']
    y_train = data_splits['y_train']
    X_val = data_splits['X_val']
    y_val = data_splits['y_val']
    X_test = data_splits['X_test']
    y_test = data_splits['y_test']
    feature_cols = data_splits['feature_cols']
    oot_keys = data_splits['oot_keys']
    split_counts = data_splits.get('split_counts', {})

    print("\n" + "="*60)
    print("Training Model 2: Gradient Boosted Trees")
    print("="*60)

    model_params = config['training_params']['model_2']

    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Training split is empty after temporal filtering. Cannot train model.")

    if len(X_val) == 0 or len(y_val) == 0:
        raise ValueError(
            "Validation split is empty after temporal filtering. "
            "Need labelled data in the validation window before training can proceed."
        )

    # Preprocessing
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)

    def safe_impute(split_name, X_split):
        if X_split is None or len(X_split) == 0:
            print(f"⚠️  Skipping imputation for {split_name} split (0 samples).")
            return None
        return imputer.transform(X_split)

    X_val_imputed = safe_impute("validation", X_val)
    X_test_imputed = safe_impute("test", X_test)

    # Preprocess OOT datasets
    X_oot_imputed = {}
    for oot_key in oot_keys:
        X_oot = data_splits[f'X_{oot_key}']
        X_oot_imputed[oot_key] = safe_impute(oot_key.upper(), X_oot)

    # Train model
    model = GradientBoostingClassifier(
        n_estimators=model_params['n_estimators'],
        learning_rate=model_params['learning_rate'],
        max_depth=model_params['max_depth'],
        min_samples_split=model_params['min_samples_split'],
        min_samples_leaf=model_params['min_samples_leaf'],
        subsample=model_params['subsample'],
        random_state=model_params['random_state'],
        verbose=0
    )

    print(f"\nTraining with hyperparameters from config:")
    for key, value in model_params.items():
        if key != 'model_type':
            print(f"  {key}: {value}")
    print()

    model.fit(X_train_imputed, y_train)

    # Evaluate
    def evaluate_split(X_imputed, y_true, split_name):
        if X_imputed is None or len(y_true) == 0:
            return None

        y_pred = model.predict(X_imputed)
        y_proba = model.predict_proba(X_imputed)[:, 1]

        return {
            'split': split_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'y_pred': y_pred,
            'y_proba': y_proba
        }

    results = {
        'train': evaluate_split(X_train_imputed, y_train, 'Training'),
        'val': evaluate_split(X_val_imputed, y_val, 'Validation'),
        'test': evaluate_split(X_test_imputed, y_test, 'Test')
    }

    # Evaluate on all OOT periods
    for oot_key in oot_keys:
        y_oot = data_splits[f'y_{oot_key}']
        results[oot_key] = evaluate_split(X_oot_imputed[oot_key], y_oot, oot_key.upper())

    # Print results
    print("\n" + "-"*60)
    print("Model 2: Gradient Boosted Trees - Evaluation Results")
    print("-"*60)
    print(f"{'Split':<15} {'Accuracy':<12} {'ROC-AUC':<12} {'Samples':<10}")
    print("-"*60)

    ordered_split_keys = ['train', 'val', 'test'] + oot_keys

    for split_key in ordered_split_keys:
        result = results.get(split_key)
        split_label = result['split'] if result else split_key.upper() if split_key in oot_keys else {
            'train': 'Training',
            'val': 'Validation',
            'test': 'Test'
        }.get(split_key, split_key.upper())
        n_samples = len(data_splits[f'y_{split_key}']) if f'y_{split_key}' in data_splits else 0

        if result:
            print(f"{split_label:<15} {result['accuracy']:<12.4f} {result['roc_auc']:<12.4f} {n_samples:<10d}")
        else:
            print(f"{split_label:<15} {'N/A':<12} {'N/A':<12} {n_samples:<10d}")

    print("-"*60)

    # Validation report
    if results['val']:
        print("\nValidation Classification Report:")
        print(classification_report(y_val, results['val']['y_pred'], target_names=['No Default', 'Default']))

    # OOT evaluation
    if any(results[oot_key] for oot_key in oot_keys):
        print("\n" + "="*60)
        print("⚠️  OUT-OF-TIME (OOT) EVALUATION")
        print("="*60)

        val_auc = results['val']['roc_auc']

        for oot_key in oot_keys:
            if results[oot_key]:
                oot_result = results[oot_key]
                oot_auc = oot_result['roc_auc']
                degradation = val_auc - oot_auc

                print(f"\n{oot_key.upper()} Results:")
                print(f"  Accuracy: {oot_result['accuracy']:.4f}")
                print(f"  ROC-AUC:  {oot_auc:.4f}")
                print(f"  Degradation from Validation: {degradation:+.4f} ({degradation*100:+.1f}%)")

                if degradation > 0.05:
                    print(f"  ⚠️  WARNING: Significant degradation!")
                elif degradation > 0.02:
                    print(f"  ⚠️  CAUTION: Moderate degradation")
                else:
                    print(f"  ✅ Performance stable")

        # Compare OOT periods
        if len(oot_keys) > 1:
            print(f"\n{'─'*60}")
            print("OOT Period Comparison (Temporal Trend Analysis):")
            print(f"{'─'*60}")

            oot_aucs = [(oot_key, results[oot_key]['roc_auc']) for oot_key in oot_keys if results[oot_key]]

            for i in range(len(oot_aucs) - 1):
                curr_key, curr_auc = oot_aucs[i]
                next_key, next_auc = oot_aucs[i + 1]
                trend = next_auc - curr_auc

                trend_symbol = "📈" if trend > 0.01 else "📉" if trend < -0.01 else "➡️"
                print(f"{curr_key.upper()} → {next_key.upper()}: {trend:+.4f} {trend_symbol}")

            if len(oot_aucs) > 0:
                first_auc = oot_aucs[0][1]
                last_auc = oot_aucs[-1][1]
                overall_trend = last_auc - first_auc

                print(f"\nOverall OOT Trend: {overall_trend:+.4f}")
                if overall_trend < -0.05:
                    print("  ⚠️  Model degrading - consider retraining")
                elif overall_trend > 0.02:
                    print("  ✅ Model improving (verify data quality)")
                else:
                    print("  ✅ Model stable across OOT periods")

    # Feature importance
    # Ensure feature_cols matches model feature_importances length
    n_importances = len(model.feature_importances_)
    n_features = len(feature_cols)

    if n_importances != n_features:
        print(f"\nWarning: Feature count mismatch - model has {n_importances} importances but feature_cols has {n_features} features")
        print("Using first N features that match importance count...")
        # Use only the features that match the importance count
        matched_feature_cols = feature_cols[:n_importances] if n_importances < n_features else feature_cols + [f'feature_{i}' for i in range(n_features, n_importances)]
    else:
        matched_feature_cols = feature_cols

    feature_importance = pd.DataFrame({
        'feature': matched_feature_cols,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    print("\nTop 20 Most Important Features:")
    print(feature_importance[['feature', 'importance']].head(20).to_string(index=False))

    metrics = {
        'model_name': 'gradient_boosting',
        'model_id': 'model_2',
        'train_accuracy': float(results['train']['accuracy']),
        'val_accuracy': float(results['val']['accuracy']),
        'train_roc_auc': float(results['train']['roc_auc']),
        'val_roc_auc': float(results['val']['roc_auc']),
        'test_accuracy': float(results['test']['accuracy']) if results['test'] else None,
        'test_roc_auc': float(results['test']['roc_auc']) if results['test'] else None,
        'feature_count': len(feature_cols),
        'train_samples': len(y_train),
        'val_samples': len(y_val),
        'test_samples': len(y_test)
    }

    # Add OOT metrics
    for oot_key in oot_keys:
        if results[oot_key]:
            metrics[f'{oot_key}_accuracy'] = float(results[oot_key]['accuracy'])
            metrics[f'{oot_key}_roc_auc'] = float(results[oot_key]['roc_auc'])
            metrics[f'{oot_key}_samples'] = len(data_splits[f'y_{oot_key}'])
        else:
            metrics[f'{oot_key}_accuracy'] = None
            metrics[f'{oot_key}_roc_auc'] = None
            metrics[f'{oot_key}_samples'] = 0

    preprocessing = {'imputer': imputer}

    return model, metrics, preprocessing, feature_importance


def save_model_artifacts(model, metrics, preprocessing, feature_importance,
                         feature_cols, label_def, snapshot_date_str, config):
    """Save model artifacts"""
    model_store_dir = "model_store/model_2/"
    os.makedirs(model_store_dir, exist_ok=True)

    joblib.dump(model, os.path.join(model_store_dir, "model.pkl"))
    joblib.dump(preprocessing, os.path.join(model_store_dir, "preprocessing.pkl"))

    with open(os.path.join(model_store_dir, "features.json"), 'w') as f:
        json.dump({'features': feature_cols}, f, indent=2)

    feature_importance.to_csv(os.path.join(model_store_dir, "feature_importance.csv"), index=False)

    metadata = {
        **metrics,
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'snapshot_date': snapshot_date_str,
        'label_definition': label_def,
        'model_type': 'Gradient Boosting Classifier',
        'temporal_splits': config['temporal_splits'],
        'hyperparameters': config['training_params']['model_2'],
        'preprocessing': {'imputation': 'median'}
    }

    with open(os.path.join(model_store_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Model 2 artifacts saved to {model_store_dir}")
    return model_store_dir


def main(snapshotdate, config_path="model_config.json"):
    print('\n\n---starting job: model_2_automl_v2---\n\n')

    # Load config (pass snapshot_date for dynamic window calculation)
    config = load_config(config_path, snapshot_date_str=snapshotdate)
    if config is None:
        print("ERROR: Config file required. Exiting.")
        return

    spark = pyspark.sql.SparkSession.builder \
        .appName("model_2_automl_v2") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    try:
        df_train, label_def = load_training_data(snapshotdate, spark, config)

        if df_train is None:
            print("Error: Could not load data.")
            spark.stop()
            return

        data_splits = prepare_ml_dataset_with_config(df_train, spark, config)

        model, metrics, preprocessing, feature_importance = train_and_evaluate_gradient_boosting(
            data_splits, config
        )

        model_store_dir = save_model_artifacts(
            model, metrics, preprocessing, feature_importance,
            data_splits['feature_cols'], label_def, snapshotdate, config
        )

        print(f"\n{'='*60}")
        print("Model 2 Training Complete")
        print(f"{'='*60}")
        print(f"Validation ROC-AUC: {metrics['val_roc_auc']:.4f}")

        # Print all OOT results
        oot_keys = data_splits['oot_keys']
        for oot_key in oot_keys:
            oot_auc = metrics.get(f'{oot_key}_roc_auc')
            if oot_auc:
                print(f"{oot_key.upper()} ROC-AUC:       {oot_auc:.4f}")

        print(f"Model Store:        {model_store_dir}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        spark.stop()

    print('\n\n---completed job: model_2_automl_v2---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model 2 with temporal config (supports multiple OOT)")
    parser.add_argument("--snapshotdate", type=str, required=True)
    parser.add_argument("--config", type=str, default="model_config.json")

    args = parser.parse_args()
    main(args.snapshotdate, args.config)