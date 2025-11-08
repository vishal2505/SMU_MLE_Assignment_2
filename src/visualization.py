"""
Visualization Script
Creates charts for model performance and stability over time
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from pipeline_config.config import (
    GOLD_PATH, MODEL_STORE_PATH, RESULTS_PATH,
    MODEL_MONITORING_FILE, MONITORING_THRESHOLDS,
    MONITOR_START_DATE, MONITOR_END_DATE
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def load_monitoring_history():
    """
    Load monitoring history from cumulative monitoring file
    
    Returns:
        DataFrame with monitoring history
    """
    monitoring_file = MODEL_STORE_PATH / MODEL_MONITORING_FILE
    
    if not monitoring_file.exists():
        print(f"No monitoring history found at: {monitoring_file}")
        return None
    
    with open(monitoring_file, 'r') as f:
        data = json.load(f)
    
    if 'monitoring_history' not in data or len(data['monitoring_history']) == 0:
        print("Monitoring history is empty")
        return None
    
    # Convert to DataFrame
    records = []
    for record in data['monitoring_history']:
        row = {
            'snapshot_date': record['snapshot_date'],
            'model_name': record['model_name'],
            'monitored_at': record['monitored_at']
        }
        row.update(record['metrics'])
        row.update({f'check_{k}': v for k, v in record['threshold_checks'].items()})
        records.append(row)
    
    df = pd.DataFrame(records)
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    df = df.sort_values('snapshot_date')

    # Filter to configured monitoring period
    try:
        start_dt = pd.to_datetime(MONITOR_START_DATE)
        end_dt = pd.to_datetime(MONITOR_END_DATE)
        before_len = len(df)
        df = df[(df['snapshot_date'] >= start_dt) & (df['snapshot_date'] <= end_dt)]
        after_len = len(df)
        if after_len == 0:
            print(f"Warning: No monitoring records within configured range {MONITOR_START_DATE} to {MONITOR_END_DATE}")
        else:
            print(f"Filtered monitoring records to configured range: {MONITOR_START_DATE} to {MONITOR_END_DATE} (kept {after_len}/{before_len})")
    except Exception as e:
        print(f"Warning: could not apply monitoring date filter: {e}")
    
    print(f"Loaded monitoring history: {len(df)} records")
    print(f"Date range: {df['snapshot_date'].min()} to {df['snapshot_date'].max()}\n")
    
    return df


def plot_performance_metrics(df, output_dir):
    """
    Plot performance metrics over time
    
    Args:
        df: DataFrame with monitoring history
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance Metrics Over Time', fontsize=16, fontweight='bold')
    
    metrics = ['auc_roc', 'accuracy', 'precision', 'recall', 'f1_score', 'log_loss']
    thresholds = {
        'auc_roc': MONITORING_THRESHOLDS['auc_roc_min'],
        'precision': MONITORING_THRESHOLDS['precision_min'],
        'recall': MONITORING_THRESHOLDS['recall_min'],
        'f1_score': MONITORING_THRESHOLDS['f1_score_min']
    }
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        
        if metric in df.columns:
            # Plot metric
            ax.plot(df['snapshot_date'], df[metric], marker='o', linewidth=2, markersize=6, label=metric.upper())
            
            # Add threshold line if applicable
            if metric in thresholds:
                ax.axhline(y=thresholds[metric], color='r', linestyle='--', linewidth=1.5, 
                          label=f'Threshold ({thresholds[metric]:.2f})')
            
            ax.set_xlabel('Date')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').upper())
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_file = output_dir / 'performance_metrics_over_time.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_psi_over_time(df, output_dir):
    """
    Plot PSI (Population Stability Index) over time
    
    Args:
        df: DataFrame with monitoring history
        output_dir: Directory to save plots
    """
    if 'psi' not in df.columns:
        print("PSI data not available, skipping PSI plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot PSI
    ax.plot(df['snapshot_date'], df['psi'], marker='o', linewidth=2, markersize=6, 
            color='navy', label='PSI')
    
    # Add threshold lines
    ax.axhline(y=MONITORING_THRESHOLDS['psi_warning'], color='orange', linestyle='--', 
              linewidth=1.5, label=f"Warning ({MONITORING_THRESHOLDS['psi_warning']})")
    ax.axhline(y=MONITORING_THRESHOLDS['psi_critical'], color='red', linestyle='--', 
              linewidth=1.5, label=f"Critical ({MONITORING_THRESHOLDS['psi_critical']})")
    
    # Add colored background zones
    ax.axhspan(0, MONITORING_THRESHOLDS['psi_warning'], alpha=0.1, color='green', label='Stable')
    ax.axhspan(MONITORING_THRESHOLDS['psi_warning'], MONITORING_THRESHOLDS['psi_critical'], 
              alpha=0.1, color='orange', label='Moderate Drift')
    ax.axhspan(MONITORING_THRESHOLDS['psi_critical'], ax.get_ylim()[1], 
              alpha=0.1, color='red', label='Significant Drift')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('PSI Value')
    ax.set_title('Population Stability Index (PSI) Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_file = output_dir / 'psi_over_time.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_confusion_matrix_trend(df, output_dir):
    """
    Plot confusion matrix metrics over time
    
    Args:
        df: DataFrame with monitoring history
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Confusion Matrix Components Over Time', fontsize=16, fontweight='bold')
    
    metrics = ['true_positives', 'false_positives', 'true_negatives', 'false_negatives']
    colors = ['green', 'orange', 'blue', 'red']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        ax = axes[idx // 2, idx % 2]
        
        if metric in df.columns:
            ax.plot(df['snapshot_date'], df[metric], marker='o', linewidth=2, 
                   markersize=6, color=color, label=metric.replace('_', ' ').title())
            ax.set_xlabel('Date')
            ax.set_ylabel('Count')
            ax.set_title(metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_file = output_dir / 'confusion_matrix_trend.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_threshold_compliance(df, output_dir):
    """
    Plot threshold compliance over time
    
    Args:
        df: DataFrame with monitoring history
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    checks = ['check_auc_roc_check', 'check_precision_check', 'check_recall_check', 'check_f1_score_check']
    check_labels = ['AUC-ROC', 'Precision', 'Recall', 'F1-Score']
    
    # Calculate compliance rate for each date
    compliance_data = []
    for _, row in df.iterrows():
        passed = sum([row.get(check, False) for check in checks])
        total = len(checks)
        compliance_rate = (passed / total) * 100
        compliance_data.append({
            'snapshot_date': row['snapshot_date'],
            'compliance_rate': compliance_rate,
            'passed_checks': passed,
            'total_checks': total
        })
    
    compliance_df = pd.DataFrame(compliance_data)
    
    # Plot compliance rate
    ax.plot(compliance_df['snapshot_date'], compliance_df['compliance_rate'], 
           marker='o', linewidth=2, markersize=8, color='darkgreen')
    ax.axhline(y=100, color='green', linestyle='--', linewidth=1.5, label='100% Compliance')
    ax.axhline(y=75, color='orange', linestyle='--', linewidth=1.5, label='75% Compliance')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Compliance Rate (%)')
    ax.set_title('Model Threshold Compliance Over Time', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    output_file = output_dir / 'threshold_compliance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_prediction_distribution(output_dir):
    """
    Plot distribution of predictions over time
    
    Args:
        output_dir: Directory to save plots
    """
    predictions_dir = GOLD_PATH / "predictions"
    prediction_files = sorted(predictions_dir.glob("predictions_*.parquet"))
    
    if not prediction_files:
        print("No prediction files found, skipping distribution plot")
        return
    
    # Sample up to 10 time periods evenly
    if len(prediction_files) > 10:
        indices = np.linspace(0, len(prediction_files) - 1, 10, dtype=int)
        prediction_files = [prediction_files[i] for i in indices]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for pred_file in prediction_files:
        df = pd.read_parquet(pred_file)
        # Extract snapshot date portion from filename tail (YYYY_MM_DD)
        stem = pred_file.stem
        # filenames like predictions_<MODEL>_YYYY_MM_DD
        parts = stem.split('_')
        # Robust: last 3 parts are date tokens
        date_token = '_'.join(parts[-3:])
        # Parse underscore format explicitly to avoid parser errors
        try:
            date_dt = datetime.strptime(date_token, '%Y_%m_%d')
        except Exception:
            # Fallback: try replacing underscores with dashes
            try:
                date_dt = pd.to_datetime(date_token.replace('_', '-'))
            except Exception:
                print(f"Warning: could not parse date from filename {pred_file.name}; skipping")
                continue
        # Filter by configured monitoring period
        start_dt = pd.to_datetime(MONITOR_START_DATE)
        end_dt = pd.to_datetime(MONITOR_END_DATE)
        if not (start_dt <= date_dt <= end_dt):
            continue
        date_str = date_dt.strftime('%Y-%m-%d')

        # Normalize to 'prediction' column for plotting
        if 'prediction' not in df.columns:
            if 'prediction_proba' in df.columns:
                df = df.rename(columns={'prediction_proba': 'prediction'})
            else:
                print(f"Warning: file {pred_file.name} has no 'prediction' or 'prediction_proba' column; skipping")
                continue

        ax.hist(df['prediction'], bins=50, alpha=0.5, label=date_str, density=True)
    
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Predictions Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'prediction_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def generate_monitoring_summary_report(df, output_dir):
    """
    Generate a text summary report
    
    Args:
        df: DataFrame with monitoring history
        output_dir: Directory to save report
    """
    report = []
    report.append("="*70)
    report.append("MODEL MONITORING SUMMARY REPORT")
    report.append("="*70)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nMonitoring Period: {df['snapshot_date'].min().strftime('%Y-%m-%d')} to {df['snapshot_date'].max().strftime('%Y-%m-%d')}")
    report.append(f"Number of Monitoring Points: {len(df)}")
    report.append(f"Model: {df['model_name'].iloc[0]}")
    
    report.append("\n" + "="*70)
    report.append("LATEST METRICS")
    report.append("="*70)
    latest = df.iloc[-1]
    report.append(f"\nSnapshot Date: {latest['snapshot_date'].strftime('%Y-%m-%d')}")
    report.append(f"AUC-ROC:     {latest['auc_roc']:.4f}")
    report.append(f"Accuracy:    {latest['accuracy']:.4f}")
    report.append(f"Precision:   {latest['precision']:.4f}")
    report.append(f"Recall:      {latest['recall']:.4f}")
    report.append(f"F1-Score:    {latest['f1_score']:.4f}")
    if 'psi' in latest:
        report.append(f"PSI:         {latest['psi']:.4f}")
    
    report.append("\n" + "="*70)
    report.append("AVERAGE METRICS")
    report.append("="*70)
    report.append(f"\nAUC-ROC:     {df['auc_roc'].mean():.4f}")
    report.append(f"Accuracy:    {df['accuracy'].mean():.4f}")
    report.append(f"Precision:   {df['precision'].mean():.4f}")
    report.append(f"Recall:      {df['recall'].mean():.4f}")
    report.append(f"F1-Score:    {df['f1_score'].mean():.4f}")
    if 'psi' in df.columns:
        report.append(f"PSI:         {df['psi'].mean():.4f}")
    
    report.append("\n" + "="*70)
    report.append("THRESHOLD COMPLIANCE")
    report.append("="*70)
    if 'check_all_passed' in df.columns:
        compliance_rate = df['check_all_passed'].mean() * 100
        report.append(f"\nOverall Compliance Rate: {compliance_rate:.1f}%")
    
    report.append("\n" + "="*70)
    
    # Save report
    output_file = output_dir / 'monitoring_summary_report.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Saved: {output_file}")
    
    # Also print to console
    print('\n'.join(report))


def main():
    """Main visualization pipeline"""
    print(f"\n{'='*70}")
    print(f"Model Monitoring Visualization")
    print(f"{'='*70}\n")
    
    # Create output directory
    output_dir = RESULTS_PATH / "monitoring_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    
    # Load monitoring history
    df = load_monitoring_history()
    
    if df is None or len(df) < 2:
        print("Insufficient monitoring data for visualization (need at least 2 records)")
        return
    
    print("Generating visualizations...\n")
    
    # Generate plots
    plot_performance_metrics(df, output_dir)
    plot_psi_over_time(df, output_dir)
    plot_confusion_matrix_trend(df, output_dir)
    plot_threshold_compliance(df, output_dir)
    plot_prediction_distribution(output_dir)
    
    # Generate summary report
    generate_monitoring_summary_report(df, output_dir)
    
    print(f"\n{'='*70}")
    print(f"Visualization completed successfully")
    print(f"All charts saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
