# ML Pipeline for Loan Default Prediction

## Overview
This project implements an end-to-end machine learning pipeline for predicting loan defaults using Apache Airflow for orchestration. The pipeline is structured as **three separate DAGs** with clear dependencies to ensure proper data flow and temporal validation.

## Architecture

### Three-DAG Structure

#### DAG 1: Data Processing Pipeline
- **File**: `dags/dag_1_data_processing.py`
- **Schedule**: Monthly (1st of month), 2023-01-01 to 2024-12-01
- **Purpose**: Process Bronze → Silver → Gold layers
- **Output**: Feature stores (MOB=0) and label stores (MOB=6)

#### DAG 2: Model Training Pipeline  
- **File**: `dags/dag_2_model_training.py`
- **Schedule**: Runs once on 2024-12-01
- **Dependencies**: Waits for DAG 1 via ExternalTaskSensor
- **Purpose**: Train models using temporal window splits
- **Output**: Trained models, model_config.json, model_evaluation.json

#### DAG 3: Inference & Monitoring Pipeline
- **File**: `dags/dag_3_inference_monitoring.py`
- **Schedule**: Manual trigger (or scheduled)
- **Dependencies**: Waits for DAG 2 via ExternalTaskSensor
- **Purpose**: Run inference on OOT data and monitor performance
- **Output**: Predictions, monitoring results, visualizations

### Data Pipeline (Bronze → Silver → Gold)
- **Bronze Layer**: Raw data ingestion from source systems
- **Silver Layer**: Data cleaning, type enforcement, and feature engineering
- **Gold Layer**: Feature store (MOB=0) and label store (MOB=6)

## Project Structure
```
.
├── pipeline_config/
│   ├── config.py                  # Central configuration file
│   └── __init__.py                # Package initialization
├── dags/
│   ├── dag_1_data_processing.py   # Data processing DAG (Bronze→Silver→Gold)
│   ├── dag_2_model_training.py    # Model training DAG (temporal windows)
│   └── dag_3_inference_monitoring.py  # Inference & monitoring DAG (OOT)
├── src/
│   ├── data_processing_bronze_table.py
│   ├── data_processing_silver_table.py
│   ├── data_processing_gold_table.py
│   ├── run_bronze_processing.py   # Bronze layer wrapper
│   ├── run_silver_processing.py   # Silver layer wrapper
│   ├── run_gold_processing.py     # Gold layer wrapper
│   ├── model_training.py          # Model training script
│   ├── model_inference.py         # Model inference script
│   ├── model_monitoring.py        # Model monitoring script
│   └── visualization.py           # Visualization generation
├── data/                          # Source data files
├── datamart/
│   ├── bronze/                    # Raw data snapshots
│   ├── silver/                    # Cleaned data
│   └── gold/                      # Feature & label stores, predictions
├── model_store/                   # Trained models and configs
├── results/                       # Monitoring visualizations
├── requirements.txt
├── Dockerfile
└── docker-compose.yaml
```

## Key Features

### 1. Three-DAG Architecture
- **DAG 1**: Data processing with monthly backfill
- **DAG 2**: Model training with temporal window validation
- **DAG 3**: OOT inference and monitoring
- **Dependencies**: ExternalTaskSensor ensures proper execution order
- **Validation**: Short-circuit operators check data and model availability

### 2. Temporal Window Configuration
Uses **absolute date ranges** to prevent data leakage:
```python
TEMPORAL_SPLITS = {
    "train": {"start_date": "2023-01-01", "end_date": "2023-12-01"},
    "validation": {"start_date": "2024-01-01", "end_date": "2024-03-01"},
    "test": {"start_date": "2024-04-01", "end_date": "2024-05-01"},
    "oot": {"start_date": "2024-06-01", "end_date": "2024-06-01"}
}
```

### 3. Temporal Leakage Prevention
- Features extracted at MOB=0 (loan application time)
- Labels extracted at MOB=6 (6 months after application)
- Fixed temporal windows (no random splitting)
- OOT data never seen during training

### 4. Model Training
- Trains 3 models: Logistic Regression, Random Forest, XGBoost
- Uses temporal window splits from configuration
- Automatically selects best model based on validation AUC-ROC
- Saves model artifacts and configuration to `model_store/`

### 5. Model Configuration Files
- **model_config.json**: Identifies the best model and selection criteria
- **model_evaluation.json**: Detailed evaluation metrics for all models
- **model_monitoring.json**: Cumulative monitoring history

### 6. Model Monitoring
- Performance Metrics: AUC-ROC, Accuracy, Precision, Recall, F1-Score
- Distribution Drift: PSI (Population Stability Index)
- Threshold Compliance: Automatic alerting when metrics fall below thresholds
- OOT Evaluation: True out-of-time validation on unseen data

### 7. Visualizations
Generated in `results/monitoring_visualizations/`:
- Performance metrics over time
- PSI trend analysis
- Confusion matrix components
- Threshold compliance rate
- Prediction distribution evolution

## Configuration (`pipeline_config/config.py`)

### Temporal Window Configuration
```python
TEMPORAL_WINDOW_MODE = "absolute"
TEMPORAL_SPLITS = {
    "train": {"start_date": "2023-01-01", "end_date": "2023-12-01"},
    "validation": {"start_date": "2024-01-01", "end_date": "2024-03-01"},
    "test": {"start_date": "2024-04-01", "end_date": "2024-05-01"},
    "oot": {"start_date": "2024-06-01", "end_date": "2024-06-01"}
}

DATA_PROCESSING_START_DATE = "2023-01-01"
DATA_PROCESSING_END_DATE = "2024-12-01"
MODEL_TRAINING_TRIGGER_DATE = "2024-12-01"
```

### Key Parameters
```python
FEATURE_MOB = 0              # Features at application time
LABEL_MOB = 6                # Labels at 6 months
DPD_THRESHOLD = 30           # Days past due for default definition

# Model configurations
MODELS = {
    'LogisticRegression': {...},
    'RandomForest': {...},
    'XGBoost': {...}
}

# Monitoring thresholds
MONITORING_THRESHOLDS = {
    'auc_roc_min': 0.70,
    'precision_min': 0.60,
    'recall_min': 0.50,
    'f1_score_min': 0.55,
    'psi_warning': 0.1,
    'psi_critical': 0.2
}
```
    'f1_score_min': 0.55,
    'psi_warning': 0.1,
    'psi_critical': 0.2
}
```

## Running the Pipeline

### Prerequisites
- Docker and Docker Compose installed
- At least 8GB RAM available

### Quick Start

1. **Build and start Docker containers**:
```bash
docker-compose build
docker-compose up -d
```

2. **Access Airflow UI**:
- URL: http://localhost:8080
- Username: airflow
- Password: airflow

3. **Run the Three DAGs in sequence**:

#### Step 1: Data Processing (DAG 1)
```
1. Enable dag_1_data_processing in Airflow UI
2. It will automatically backfill from 2023-01-01 to 2024-12-01 (24 monthly runs)
3. Wait for all runs to complete
4. Verify: ls datamart/gold/feature_store/ and datamart/gold/label_store/
```

#### Step 2: Model Training (DAG 2)
```
1. After DAG 1 completes through 2024-06-01
2. Enable dag_2_model_training
3. It will automatically start when dependencies are met (runs on 2024-12-01)
4. Training uses temporal windows: train/validation/test
5. Verify: cat model_store/model_config.json
```

#### Step 3: Inference & Monitoring (DAG 3)
```
1. After DAG 2 completes
2. Enable dag_3_inference_monitoring
3. Manually trigger the DAG
4. It runs inference on OOT data (2024-06-01)
5. Verify: ls datamart/gold/predictions/ and results/monitoring_visualizations/
```

## DAG Workflow

### Three-DAG Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│ DAG 1: Data Processing (Monthly: 2023-01 to 2024-12)       │
├─────────────────────────────────────────────────────────────┤
│   start_pipeline                                            │
│        ↓                                                    │
│   process_bronze_tables                                     │
│        ↓                                                    │
│   process_silver_tables                                     │
│        ↓                                                    │
│   process_gold_tables                                       │
│        ↓                                                    │
│   data_processing_complete                                  │
└────────────────────┬────────────────────────────────────────┘
                     │ All 24 monthly runs complete
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ DAG 2: Model Training (Once: 2024-12-01)                   │
├─────────────────────────────────────────────────────────────┤
│   wait_for_data_processing (ExternalTaskSensor → DAG 1)    │
│        ↓                                                    │
│   check_all_data_processed (validates 2023-01 to 2024-06)  │
│        ↓                                                    │
│   train_models (temporal: train/val/test splits)           │
│        ↓                                                    │
│   training_complete                                         │
└────────────────────┬────────────────────────────────────────┘
                     │ Models trained & saved
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ DAG 3: Inference & Monitoring (Manual trigger)             │
├─────────────────────────────────────────────────────────────┤
│   wait_for_model_training (ExternalTaskSensor → DAG 2)     │
│        ↓                                                    │
│   check_model_trained (validates model_config.json exists) │
│        ↓                                                    │
│   check_oot_data_available (validates OOT feature/labels)  │
│        ↓                                                    │
│   run_inference_oot (predict on 2024-06-01)               │
│        ↓                                                    │
│   run_monitoring_oot (evaluate OOT performance)            │
│        ↓                                                    │
│   generate_visualizations (charts & reports)               │
│        ↓                                                    │
│   pipeline_complete                                         │
└─────────────────────────────────────────────────────────────┘
```

### Key Features
- **ExternalTaskSensor**: Ensures DAG dependencies are met
- **Data Validation**: Short-circuit operators check prerequisites
- **Temporal Windows**: Fixed date ranges prevent data leakage
- **OOT Validation**: True out-of-time testing on 2024-06-01

## Model Store Structure

```
model_store/
├── model_config.json              # Best model selection
├── model_evaluation.json          # All models evaluation
├── model_monitoring.json          # Monitoring history
├── LogisticRegression/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── metadata.json
├── RandomForest/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── metadata.json
└── XGBoost/
    ├── model.pkl
    ├── scaler.pkl
    └── metadata.json
```

## Monitoring Outputs

### Performance Metrics
Tracked for each prediction batch:
- AUC-ROC: Area under ROC curve
- Accuracy: Overall correctness
- Precision: Positive predictive value
- Recall: True positive rate
- F1-Score: Harmonic mean of precision and recall
- Log Loss: Probabilistic error

### Distribution Drift (PSI)
- **< 0.1**: No significant change (OK)
- **0.1 - 0.2**: Moderate change (WARNING)
- **> 0.2**: Significant change (CRITICAL - consider retraining)

## Model Governance & SOP

### Retraining Triggers
1. **Performance Degradation**: Any metric falls below threshold for 2+ consecutive periods
2. **Distribution Drift**: PSI > 0.2 (CRITICAL level)
3. **Scheduled Retraining**: Monthly with rolling 12-month window

### Deployment Options
1. **Shadow Mode**: Run new model in parallel, compare results
2. **Canary Deployment**: Gradually shift traffic to new model
3. **Blue-Green Deployment**: Switch entirely after validation

### Monitoring Frequency
- **Real-time**: Predictions are made for each new snapshot
- **Batch Evaluation**: Monthly after labels become available (6-month lag)
- **Reporting**: Automated visualizations generated after each monitoring run

## Troubleshooting

### Common Issues

1. **DAG 2 not starting**:
   - Check if DAG 1 has completed for all dates through 2024-06-01
   - Verify feature and label stores exist: `ls datamart/gold/feature_store/` and `ls datamart/gold/label_store/`
   - Check ExternalTaskSensor status in Airflow UI

2. **DAG 3 not starting**:
   - Ensure DAG 2 has completed successfully
   - Verify model_config.json exists: `cat model_store/model_config.json`
   - Check OOT data availability for 2024-06-01

3. **Models not training**:
   - Check if sufficient data exists (need data from 2023-01-01 to 2024-05-01)
   - Verify temporal window configuration in `pipeline_config/config.py`
   - Check logs: `docker-compose logs webserver`

4. **Inference skipped**:
   - Models must be trained first (DAG 2 must complete)
   - Check `model_config.json` exists in `model_store/`
   - Verify OOT feature store exists for 2024-06-01

5. **Monitoring skipped**:
   - Requires both predictions and labels for same date
   - Labels are created 6 months after features
   - Check both exist: `ls datamart/gold/predictions/` and `ls datamart/gold/label_store/`

6. **Memory errors**:
   - Increase Docker memory allocation (8GB minimum)
   - Reduce Spark driver memory in `pipeline_config/config.py`

7. **Import errors**:
   - Ensure `pipeline_config/` folder (not `config/`) is used
   - Verify volume mounts in docker-compose.yaml
   - Restart containers: `docker-compose down && docker-compose up -d`

## Dependencies

See `requirements.txt`:
- pandas, numpy: Data manipulation
- scikit-learn: ML models and metrics
- xgboost: Gradient boosting
- pyspark: Distributed data processing
- apache-airflow: Workflow orchestration
- matplotlib, seaborn: Visualizations

## Contact

For questions or issues, contact the ML team at ml-team@company.com

---

**Note**: This pipeline follows best practices for production ML systems including:
- **Three-DAG architecture** with clear separation of concerns
- **Temporal validation** with fixed date ranges to prevent data leakage
- **ExternalTaskSensor** for proper DAG dependencies
- **Data validation checks** before each critical step
- **OOT evaluation** on truly unseen data (2024-06-01)
- **Automated model selection** and versioning
- **Continuous monitoring** with drift detection (PSI)
- **Clear model governance** procedures
