"""
Model Monitoring Script
Evaluates model predictions against actual labels
Calculates performance metrics and PSI for drift detection
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, log_loss, confusion_matrix
)

# --- JSON Sanitization Helper -------------------------------------------------
# The monitoring output dictionaries can contain numpy scalar types (e.g. np.bool_)
# produced by threshold comparisons or metric calculations. Python's json module
# cannot serialize numpy scalar types directly and raises TypeError. We provide a
# small recursive sanitizer to convert these to native Python types before dumping.
def _sanitize_for_json(obj):
    """Recursively convert numpy/pandas types to JSON-serializable native types.

    Handles:
      - numpy scalar types (bool_, integer, floating)
      - pandas Timestamp -> ISO string
      - dict / list / tuple / set containers
    """
    import numpy as _np
    import pandas as _pd
    from datetime import datetime as _dt

    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_pd.Timestamp, _dt)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]
    return obj

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from pipeline_config.config import (
    GOLD_PATH, MODEL_STORE_PATH, MONITORING_THRESHOLDS,
    MODEL_MONITORING_FILE, MONITORING_METRICS, LABEL_MOB
)


def load_predictions(snapshot_date):
    """Load predictions for a specific snapshot date with flexible filename handling.

    Supports both legacy naming (predictions_<DATE>.parquet) and new naming
    (predictions_<MODEL_NAME>_<DATE>.parquet). Also normalizes columns so that
    a 'prediction' probability column is always available for downstream code.
    """
    date_tag = snapshot_date.replace('-', '_')
    pred_dir = GOLD_PATH / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # 1) Legacy file name
    candidates = [pred_dir / f"predictions_{date_tag}.parquet"]

    # 2) Best-model-aware name
    try:
        cfg_path = MODEL_STORE_PATH / "model_config.json"
        if cfg_path.exists():
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
            best_model = cfg.get('best_model')
            if best_model:
                candidates.append(pred_dir / f"predictions_{best_model}_{date_tag}.parquet")
    except Exception:
        pass

    # 3) Wildcard search as last resort
    if not any(p.exists() for p in candidates):
        wildcard = sorted(pred_dir.glob(f"predictions_*_{date_tag}.parquet"))
        if wildcard:
            candidates.append(wildcard[0])

    # Pick the first existing candidate
    chosen = next((p for p in candidates if isinstance(p, Path) and p.exists()), None)
    if chosen is None:
        raise FileNotFoundError(f"Predictions not found for {snapshot_date}. Looked for: "
                                f"{[str(p) for p in candidates if isinstance(p, Path)]}")

    df = pd.read_parquet(chosen)

    # Normalize columns
    if 'prediction' not in df.columns:
        # Accept new schema 'prediction_proba'
        if 'prediction_proba' in df.columns:
            df = df.rename(columns={'prediction_proba': 'prediction'})
        else:
            raise KeyError("Predictions file does not contain 'prediction' or 'prediction_proba' column")

    print(f"Loaded predictions: {chosen.name}")
    print(f"Number of predictions: {len(df)}\n")
    return df


def load_labels(snapshot_date):
    """
    Load labels aligned to the inference cohort using LABEL_MOB offset.

    For predictions generated on snapshot_date using MOB=0 features, the true
    labels are observed at snapshot_date + LABEL_MOB months.
    """
    # Compute aligned label date
    base_dt = pd.to_datetime(snapshot_date)
    aligned_dt = (base_dt + pd.DateOffset(months=LABEL_MOB)).to_pydatetime()
    label_tag = aligned_dt.strftime("%Y_%m_%d")
    label_file = GOLD_PATH / "label_store" / f"label_store_{label_tag}.parquet"

    if not label_file.exists():
        raise FileNotFoundError(
            f"Labels not found for aligned date (snapshot_date + {LABEL_MOB} months): {label_file}"
        )

    df = pd.read_parquet(label_file)
    print(f"Loaded labels (aligned by +{LABEL_MOB} months): {label_file.name}")
    print(f"Number of labels: {len(df)}\n")

    return df


def merge_predictions_labels(predictions_df, labels_df):
    """
    Merge predictions with actual labels
    
    Args:
        predictions_df: DataFrame with predictions
        labels_df: DataFrame with labels
    
    Returns:
        Merged DataFrame
    """
    # Prefer joining on both identifiers if present
    join_keys = []
    for key in ['loan_id', 'Customer_ID']:
        if key in predictions_df.columns and key in labels_df.columns:
            join_keys.append(key)
    if not join_keys:
        join_keys = ['loan_id'] if 'loan_id' in predictions_df.columns and 'loan_id' in labels_df.columns else []
    if not join_keys:
        raise KeyError("Neither 'loan_id' nor 'Customer_ID' available in both predictions and labels for joining.")

    merged_df = predictions_df.merge(
        labels_df[join_keys + ['label']],
        on=join_keys,
        how='inner'
    )
    
    print(f"Merged predictions and labels: {len(merged_df)} samples")
    if len(merged_df) == 0:
        print("WARNING: No overlap between predictions and labels on keys:", join_keys)
        # Diagnostics
        for key in join_keys:
            pred_keys = set(predictions_df[key].dropna().astype(str).head(5).values.tolist())
            lab_keys = set(labels_df[key].dropna().astype(str).head(5).values.tolist())
            print(f"  Sample {key} in predictions: {list(pred_keys)}")
            print(f"  Sample {key} in labels:      {list(lab_keys)}\n")
    else:
        print(f"Actual default rate: {merged_df['label'].mean():.2%}")
        print(f"Predicted default rate (>0.5): {(merged_df['prediction'] > 0.5).mean():.2%}\n")
    
    return merged_df


def calculate_performance_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate performance metrics
    
    Args:
        y_true: Actual labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Handle empty or single-class cases gracefully
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred) if len(y_true) else float('nan'),
        'precision': precision_score(y_true, y_pred, zero_division=0) if len(y_true) else float('nan'),
        'recall': recall_score(y_true, y_pred, zero_division=0) if len(y_true) else float('nan'),
        'f1_score': f1_score(y_true, y_pred, zero_division=0) if len(y_true) else float('nan'),
        'threshold': threshold
    }
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
    except Exception:
        metrics['auc_roc'] = float('nan')
    try:
        metrics['log_loss'] = log_loss(y_true, y_pred_proba)
    except Exception:
        metrics['log_loss'] = float('nan')
    
    # Confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        })
    except Exception:
        metrics.update({
            'true_negatives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'true_positives': 0
        })
    
    return metrics


def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index (PSI) for drift detection
    
    PSI measures the shift in distribution of predictions between training and production
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate change, investigation recommended
    PSI >= 0.2: Significant change, model may need retraining
    
    Args:
        expected: Baseline predictions (e.g., from training)
        actual: Current predictions
        bins: Number of bins for bucketing
    
    Returns:
        PSI value
    """
    # Create bins
    breakpoints = np.linspace(0, 1, bins + 1)
    breakpoints[0] = -0.001  # Include 0
    breakpoints[-1] = 1.001  # Include 1
    
    # Calculate expected distribution
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    
    # Calculate actual distribution
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    # Calculate PSI
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    
    return float(psi)


def load_baseline_predictions():
    """
    Load baseline predictions (from first monitoring period)
    Used for PSI calculation
    
    Returns:
        Array of baseline predictions or None if not available
    """
    predictions_dir = GOLD_PATH / "predictions"
    prediction_files = sorted(predictions_dir.glob("predictions_*.parquet"))
    
    if not prediction_files:
        return None
    
    # Use first predictions file as baseline
    baseline_file = prediction_files[0]
    baseline_df = pd.read_parquet(baseline_file)
    # Normalize to 'prediction'
    if 'prediction' not in baseline_df.columns and 'prediction_proba' in baseline_df.columns:
        baseline_df = baseline_df.rename(columns={'prediction_proba': 'prediction'})

    print(f"Using baseline predictions from: {baseline_file.name}")
    
    return baseline_df['prediction'].values


def check_thresholds(metrics):
    """
    Check if metrics meet the defined thresholds
    
    Args:
        metrics: Dictionary of calculated metrics
    
    Returns:
        Dictionary with threshold check results
    """
    checks = {
        'auc_roc_check': metrics['auc_roc'] >= MONITORING_THRESHOLDS['auc_roc_min'],
        'precision_check': metrics['precision'] >= MONITORING_THRESHOLDS['precision_min'],
        'recall_check': metrics['recall'] >= MONITORING_THRESHOLDS['recall_min'],
        'f1_score_check': metrics['f1_score'] >= MONITORING_THRESHOLDS['f1_score_min'],
        'psi_warning': metrics.get('psi', 0) < MONITORING_THRESHOLDS['psi_warning'],
        'psi_critical': metrics.get('psi', 0) < MONITORING_THRESHOLDS['psi_critical']
    }
    
    checks['all_passed'] = all([
        checks['auc_roc_check'],
        checks['precision_check'],
        checks['recall_check'],
        checks['f1_score_check']
    ])
    
    if metrics.get('psi'):
        if not checks['psi_critical']:
            checks['alert_level'] = 'CRITICAL'
        elif not checks['psi_warning']:
            checks['alert_level'] = 'WARNING'
        else:
            checks['alert_level'] = 'OK'
    else:
        checks['alert_level'] = 'NO_PSI'
    
    return checks


def save_monitoring_results(metrics, threshold_checks, snapshot_date, model_name):
    """
    Save monitoring results to gold layer
    
    Args:
        metrics: Dictionary of metrics
        threshold_checks: Dictionary of threshold check results
        snapshot_date: Snapshot date
        model_name: Name of the model
    """
    monitoring_result = {
        'snapshot_date': snapshot_date,
        'model_name': model_name,
        'monitored_at': datetime.now().isoformat(),
        'metrics': metrics,
        'threshold_checks': threshold_checks,
        'thresholds': MONITORING_THRESHOLDS
    }
    
    # Save to monitoring directory
    monitoring_dir = GOLD_PATH / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = snapshot_date.replace('-', '_')
    output_file = monitoring_dir / f"monitoring_{model_name}_{date_str}.json"
    
    # Sanitize before serialization
    monitoring_result_native = _sanitize_for_json(monitoring_result)
    with open(output_file, 'w') as f:
        json.dump(monitoring_result_native, f, indent=4)
    
    print(f"\n{'='*70}")
    print(f"Monitoring results saved to: {output_file}")
    print(f"{'='*70}\n")
    
    # Also append to cumulative monitoring file
    cumulative_file = MODEL_STORE_PATH / MODEL_MONITORING_FILE
    
    # Load existing monitoring history
    if cumulative_file.exists():
        with open(cumulative_file, 'r') as f:
            monitoring_history = json.load(f)
    else:
        monitoring_history = {'monitoring_history': []}
    
    # Add new result
    monitoring_history['monitoring_history'].append(monitoring_result_native)
    
    # Save updated history
    with open(cumulative_file, 'w') as f:
        json.dump(_sanitize_for_json(monitoring_history), f, indent=4)
    
    print(f"Updated cumulative monitoring: {cumulative_file}\n")

    # Optional: save parquet summary for easy querying
    try:
        import pandas as _pd
        flat = {**{f"metric_{k}": v for k, v in metrics.items()},
                **{f"check_{k}": v for k, v in threshold_checks.items()},
                'snapshot_date': snapshot_date,
                'model_name': model_name,
                'monitored_at': monitoring_result['monitored_at']}
        _pdf = _pd.DataFrame([flat])
        parquet_path = monitoring_dir / f"monitoring_{model_name}_{date_str}.parquet"
        _pdf.to_parquet(parquet_path, index=False)
        print(f"Saved monitoring parquet: {parquet_path}")
    except Exception as _e:
        print(f"Warning: could not write parquet summary: {_e}")


def print_monitoring_summary(metrics, threshold_checks):
    """Print monitoring summary"""
    print(f"\n{'='*70}")
    print("MONITORING SUMMARY")
    print(f"{'='*70}\n")
    
    print("Performance Metrics:")
    print(f"  AUC-ROC:     {metrics['auc_roc']:.4f} {'✓' if threshold_checks['auc_roc_check'] else '✗ BELOW THRESHOLD'}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f} {'✓' if threshold_checks['precision_check'] else '✗ BELOW THRESHOLD'}")
    print(f"  Recall:      {metrics['recall']:.4f} {'✓' if threshold_checks['recall_check'] else '✗ BELOW THRESHOLD'}")
    print(f"  F1-Score:    {metrics['f1_score']:.4f} {'✓' if threshold_checks['f1_score_check'] else '✗ BELOW THRESHOLD'}")
    print(f"  Log Loss:    {metrics['log_loss']:.4f}")
    
    if 'psi' in metrics:
        print(f"\nDistribution Stability:")
        print(f"  PSI:         {metrics['psi']:.4f} - {threshold_checks['alert_level']}")
        print(f"    < 0.1:     No significant change")
        print(f"    0.1-0.2:   Moderate change")
        print(f"    > 0.2:     Significant change - consider retraining")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    
    print(f"\nOverall Status: {'✓ PASSED' if threshold_checks['all_passed'] else '✗ FAILED'}")
    print(f"Alert Level: {threshold_checks['alert_level']}")
    print(f"{'='*70}\n")


def main():
    """Main monitoring pipeline"""
    parser = argparse.ArgumentParser(description='Monitor model performance')
    parser.add_argument('--snapshot_date', type=str, help='Single snapshot date (YYYY-MM-DD)')
    parser.add_argument('--start_date', type=str, help='Start date inclusive (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='End date inclusive (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"ML Model Monitoring Pipeline")
    print(f"{'='*70}\n")
    # Determine mode
    if not args.snapshot_date and not (args.start_date and args.end_date):
        raise SystemExit("Provide either --snapshot_date or both --start_date and --end_date")

    def month_range(start_date: str, end_date: str):
        sd = pd.to_datetime(start_date)
        ed = pd.to_datetime(end_date)
        cur = sd.replace(day=1)
        end_marker = ed.replace(day=1)
        while cur <= end_marker:
            yield cur.strftime('%Y-%m-%d')
            year = cur.year + (cur.month // 12)
            month = 1 if cur.month == 12 else cur.month + 1
            cur = cur.replace(year=year, month=month)

    dates = [args.snapshot_date] if args.snapshot_date else list(month_range(args.start_date, args.end_date))
    print(f"Monitoring over {len(dates)} month(s): {dates[0]} ... {dates[-1]}\n")

    baseline_predictions = load_baseline_predictions()
    successes = 0
    for dt in dates:
        print(f"--- Monitoring for {dt} ---")
        try:
            predictions_df = load_predictions(dt)
        except FileNotFoundError as e:
            print(f"[Skip] predictions missing: {e}")
            continue
        try:
            labels_df = load_labels(dt)
        except FileNotFoundError as e:
            print(f"[Skip] labels missing for monitoring date {dt}: {e}")
            # We can still record distribution-only metrics
            metrics = {
                'auc_roc': float('nan'),
                'accuracy': float('nan'),
                'precision': float('nan'),
                'recall': float('nan'),
                'f1_score': float('nan'),
                'log_loss': float('nan'),
                'true_negatives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'true_positives': 0,
                'threshold': 0.5
            }
            if baseline_predictions is not None and len(baseline_predictions) > 0:
                print("Calculating PSI (labels unavailable)...")
                psi = calculate_psi(baseline_predictions, predictions_df['prediction'].values)
                metrics['psi'] = psi
                print(f"PSI: {psi:.4f}\n")
            threshold_checks = check_thresholds(metrics)
            print_monitoring_summary(metrics, threshold_checks)
            model_name = predictions_df['model_name'].iloc[0] if 'model_name' in predictions_df.columns else 'Unknown'
            save_monitoring_results(metrics, threshold_checks, dt, model_name)
            continue

        merged_df = merge_predictions_labels(predictions_df, labels_df)
        print("Calculating performance metrics...")
        metrics = calculate_performance_metrics(
            merged_df['label'].values,
            merged_df['prediction'].values
        )
        if baseline_predictions is not None and len(baseline_predictions) > 0:
            print("Calculating PSI for distribution drift...")
            psi = calculate_psi(baseline_predictions, merged_df['prediction'].values)
            metrics['psi'] = psi
            print(f"PSI: {psi:.4f}\n")
        else:
            print("No baseline predictions available for PSI calculation\n")
        threshold_checks = check_thresholds(metrics)
        print_monitoring_summary(metrics, threshold_checks)
        model_name = predictions_df['model_name'].iloc[0] if 'model_name' in predictions_df.columns else 'Unknown'
        save_monitoring_results(metrics, threshold_checks, dt, model_name)
        successes += 1

    print(f"Monitoring pipeline completed. Successful monitored months with labels: {successes}/{len(dates)}\n")


if __name__ == "__main__":
    main()
