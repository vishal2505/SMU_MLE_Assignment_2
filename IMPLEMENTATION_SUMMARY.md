# Assignment 2 - ML Pipeline Implementation Summary

## Pipeline Architecture: Three Separate DAGs

The pipeline has been restructured into **three separate, dependent DAGs** to ensure proper sequencing, data availability checks, and temporal validation:

### DAG 1: Data Processing Pipeline (`dag_1_data_processing.py`)
- **Purpose**: Process raw data through Bronze → Silver → Gold layers
- **Schedule**: Monthly (1st of month) from 2023-01-01 to 2024-12-01
- **Catchup**: Enabled (backfills all historical data)
- **Tasks**: Bronze ingestion → Silver cleaning → Gold feature/label creation
- **Output**: Feature stores and label stores for all dates

### DAG 2: Model Training Pipeline (`dag_2_model_training.py`)
- **Purpose**: Train ML models using temporal window splits
- **Schedule**: Runs once on 2024-12-01
- **Dependencies**: Waits for DAG 1 via ExternalTaskSensor
- **Validation**: Checks all required data exists (2023-01-01 to 2024-06-01)
- **Training**: Uses fixed temporal windows (no random splitting):
  - Train: 2023-01-01 to 2023-12-01
  - Validation: 2024-01-01 to 2024-03-01
  - Test: 2024-04-01 to 2024-05-01
- **Output**: Trained models, model_config.json, model_evaluation.json

### DAG 3: Inference & Monitoring Pipeline (`dag_3_inference_monitoring.py`)
- **Purpose**: Run inference and monitoring on OOT (Out-of-Time) data
- **Schedule**: Manual trigger (or scheduled)
- **Dependencies**: Waits for DAG 2 via ExternalTaskSensor
- **Validation**: Checks model exists and OOT data available
- **OOT Period**: 2024-06-01 (truly unseen data)
- **Tasks**: Inference → Monitoring → Visualization
- **Output**: Predictions, monitoring results, performance charts

### Execution Flow
```
DAG 1: Data Processing (2023-01 to 2024-12)
    ↓ [Backfill 24 months]
    ↓ [Creates all feature & label stores]
    ↓
    ✓ Data processing complete
    ↓
DAG 2: Model Training (2024-12-01 only)
    ↓ [ExternalTaskSensor waits for DAG 1]
    ↓ [Checks all data available]
    ↓ [Trains on temporal windows]
    ↓
    ✓ Models trained & saved
    ↓
DAG 3: Inference & Monitoring (Manual)
    ↓ [ExternalTaskSensor waits for DAG 2]
    ↓ [Runs inference on OOT: 2024-06-01]
    ↓ [Monitors OOT performance]
    ↓
    ✓ Pipeline complete
```

## Completed Components

### 1. Configuration Management (`pipeline_config/config.py`)
✓ Centralized configuration for all pipeline parameters
✓ Temporal window configuration (TEMPORAL_SPLITS with absolute dates)
✓ Model hyperparameters for Logistic Regression, Random Forest, and XGBoost
✓ Feature definitions (45+ features)
✓ Monitoring thresholds (AUC, Precision, Recall, F1, PSI)
✓ Path configurations for all data layers
✓ Data processing date ranges and model training trigger date

### 2. Model Training (`src/model_training.py`)
✓ Trains 3 models: Logistic Regression, Random Forest, XGBoost
✓ Uses temporal window splits (absolute dates from config)
✓ Automatic best model selection based on validation AUC-ROC
✓ Saves model artifacts (model.pkl, scaler.pkl, metadata.json) for each model
✓ Generates `model_config.json` with best model identification
✓ Generates `model_evaluation.json` with detailed metrics for all models
✓ Handles class imbalance with balanced weights
✓ Comprehensive evaluation metrics
✓ Command-line flag: `--use_temporal_splits` for fixed temporal windows

### 3. Model Inference (`src/model_inference.py`)
✓ Loads best model from `model_config.json`
✓ Processes feature store data for predictions
✓ Handles missing features gracefully
✓ Saves predictions to gold layer with metadata
✓ Includes loan_id, customer_id, and model name tracking

### 4. Model Monitoring (`src/model_monitoring.py`)
✓ Evaluates predictions against actual labels
✓ Calculates performance metrics:
  - AUC-ROC, Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix components
  - Log Loss
✓ Calculates PSI (Population Stability Index) for drift detection
✓ Threshold compliance checking
✓ Saves individual monitoring results per snapshot
✓ Maintains cumulative `model_monitoring.json` history

### 5. Visualization (`src/visualization.py`)
✓ Performance metrics over time chart
✓ PSI trend analysis with warning/critical zones
✓ Confusion matrix components trend
✓ Threshold compliance tracking
✓ Prediction distribution evolution
✓ Summary report generation

### 6. Airflow DAGs
✓ **DAG 1: Data Processing** (`dags/dag_1_data_processing.py`)
  - Bronze → Silver → Gold layer processing
  - Monthly schedule with catchup (backfills historical data)
  - Runs from 2023-01-01 to 2024-12-01
  
✓ **DAG 2: Model Training** (`dags/dag_2_model_training.py`)
  - Waits for DAG 1 completion via ExternalTaskSensor
  - Checks all required data availability before training
  - Trains models using temporal window splits
  - Runs once on 2024-12-01
  
✓ **DAG 3: Inference & Monitoring** (`dags/dag_3_inference_monitoring.py`)
  - Waits for DAG 2 completion via ExternalTaskSensor
  - Validates model and OOT data availability
  - Runs inference on OOT period (2024-06-01)
  - Performs monitoring and generates visualizations
  - Manual trigger or scheduled execution

### 7. Docker Configuration
✓ Dockerfile with Airflow, PySpark, Java
✓ docker-compose.yaml with all necessary volumes
✓ Volume mounts for: dags, src, pipeline_config, data, datamart, model_store, results
✓ Proper Python package configuration to avoid module conflicts

### 8. Documentation
✓ Comprehensive README_PIPELINE.md
✓ Project structure documentation
✓ Configuration guide
✓ Troubleshooting section
✓ Model governance and SOP

## Key Features Implemented

### Model Store Structure
```
model_store/
├── model_config.json              # Best model selection
├── model_evaluation.json          # All models evaluation summary
├── model_monitoring.json          # Cumulative monitoring history
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

### Config JSON Files

#### model_config.json
```json
{
    "best_model": "XGBoost",
    "selection_metric": "auc_roc",
    "selection_score": 0.8523,
    "all_models": {
        "LogisticRegression": {...},
        "RandomForest": {...},
        "XGBoost": {...}
    },
    "updated_at": "2024-12-01T10:30:00"
}
```

#### model_evaluation.json
```json
{
    "evaluation_date": "2024-12-01T10:30:00",
    "models": {
        "LogisticRegression": {
            "train_metrics": {...},
            "val_metrics": {...},
            "params": {...}
        },
        ...
    }
}
```

#### model_monitoring.json
```json
{
    "monitoring_history": [
        {
            "snapshot_date": "2024-01-01",
            "model_name": "XGBoost",
            "monitored_at": "2024-07-01T10:00:00",
            "metrics": {
                "auc_roc": 0.85,
                "precision": 0.72,
                "recall": 0.68,
                "f1_score": 0.70,
                "psi": 0.08
            },
            "threshold_checks": {...},
            "thresholds": {...}
        },
        ...
    ]
}
```

## Temporal Leakage Prevention

The pipeline carefully avoids temporal leakage:

1. **Features**: Extracted at MOB=0 (loan application time)
   - Only uses information available at application
   - Excludes future payment behavior
   - Excludes loan performance metrics

2. **Labels**: Extracted at MOB=6 (6 months after application)
   - Defines default as DPD >= 30 days at month 6
   - Ensures 6-month observation period

3. **Training**: Uses historical data with proper time windows
   - 12-month rolling training window
   - Validation and test sets maintain temporal order

## Model Monitoring Strategy

### Performance Metrics
- **AUC-ROC**: Primary model selection metric (threshold: 0.70)
- **Precision**: Positive predictive value (threshold: 0.60)
- **Recall**: True positive rate (threshold: 0.50)
- **F1-Score**: Balanced metric (threshold: 0.55)

### Distribution Drift (PSI)
- **< 0.1**: Stable - No action needed
- **0.1 - 0.2**: Warning - Monitor closely
- **> 0.2**: Critical - Consider retraining

### Retraining Triggers
1. Performance degradation below thresholds for 2+ periods
2. PSI exceeds critical threshold (0.2)
3. Scheduled monthly retraining with rolling window

## How to Run

### 1. Build Docker Containers
```bash
docker-compose build
```

### 2. Start Airflow
```bash
docker-compose up -d
```

### 3. Access Airflow UI
- URL: http://localhost:8080
- Username: airflow
- Password: airflow

### 4. Run the Pipeline

#### Step 1: Start Data Processing (DAG 1)
1. Navigate to Airflow UI
2. Enable `dag_1_data_processing`
3. It will automatically backfill from 2023-01-01 to 2024-12-01 (24 monthly runs)
4. Wait for all runs to complete

#### Step 2: Trigger Model Training (DAG 2)
1. After DAG 1 completes through 2024-06-01
2. Enable `dag_2_model_training`
3. It will automatically start when dependencies are met
4. Training happens once on 2024-12-01 execution date

#### Step 3: Run Inference and Monitoring (DAG 3)
1. After DAG 2 completes successfully
2. Enable `dag_3_inference_monitoring`
3. Manually trigger the DAG
4. It will run inference and monitoring on OOT data (2024-06-01)

### Monitoring Progress

#### Check DAG 1 Progress
```bash
# Verify feature and label stores are created
ls -la datamart/gold/feature_store/
ls -la datamart/gold/label_store/
```

#### Check DAG 2 Status
```bash
# Verify model training completed
ls -la model_store/
cat model_store/model_config.json
```

#### Check DAG 3 Results
```bash
# Verify predictions and monitoring results
ls -la datamart/gold/predictions/
ls -la datamart/gold/monitoring/
ls -la results/monitoring_visualizations/
```

## Pipeline Execution Flow

### Three-DAG Architecture

```
┌─────────────────────────────────────────────────────────┐
│ DAG 1: Data Processing (Monthly: 2023-01 to 2024-12)   │
├─────────────────────────────────────────────────────────┤
│  Bronze Layer → Silver Layer → Gold Layer               │
│  - Raw ingestion                                        │
│  - Data cleaning                                        │
│  - Feature store (MOB=0) + Label store (MOB=6)         │
└────────────────────┬────────────────────────────────────┘
                     │ All data processed
                     ↓
┌─────────────────────────────────────────────────────────┐
│ DAG 2: Model Training (Once: 2024-12-01)               │
├─────────────────────────────────────────────────────────┤
│  wait_for_data_processing (ExternalTaskSensor)         │
│         ↓                                               │
│  check_all_data_processed (validate availability)      │
│         ↓                                               │
│  train_models (temporal windows: train/val/test)       │
│         ↓                                               │
│  Save: model_config.json, model_evaluation.json        │
└────────────────────┬────────────────────────────────────┘
                     │ Models trained
                     ↓
┌─────────────────────────────────────────────────────────┐
│ DAG 3: Inference & Monitoring (Manual trigger)         │
├─────────────────────────────────────────────────────────┤
│  wait_for_model_training (ExternalTaskSensor)          │
│         ↓                                               │
│  check_model_trained (validate model exists)           │
│         ↓                                               │
│  check_oot_data_available (validate OOT data)          │
│         ↓                                               │
│  run_inference_oot (predict on 2024-06-01)            │
│         ↓                                               │
│  run_monitoring_oot (evaluate OOT performance)         │
│         ↓                                               │
│  generate_visualizations (charts & reports)            │
└─────────────────────────────────────────────────────────┘
```

### Temporal Window Configuration

The pipeline uses **absolute date ranges** for temporal validation:

```python
TEMPORAL_SPLITS = {
    "train": {
        "start_date": "2023-01-01",
        "end_date": "2023-12-01"
    },
    "validation": {
        "start_date": "2024-01-01",
        "end_date": "2024-03-01"
    },
    "test": {
        "start_date": "2024-04-01",
        "end_date": "2024-05-01"
    },
    "oot": {
        "start_date": "2024-06-01",
        "end_date": "2024-06-01"
    }
}
```

### Key Advantages

1. **Clear Separation**: Data processing, training, and inference are independent
2. **Proper Dependencies**: ExternalTaskSensor ensures correct execution order
3. **Data Validation**: Short-circuit checks prevent incomplete runs
4. **Temporal Integrity**: Fixed date ranges prevent data leakage
5. **True OOT Validation**: OOT data (2024-06-01) is never used in training

## Success Criteria

✓ **Task 1: Build end-to-end ML pipeline** (5 marks)
  - ✓ Train ML models with multiple algorithms
  - ✓ Select and store best model
  - ✓ Make predictions using best model
  - ✓ Monitor performance and stability
  - ✓ Visualize results over time
  - ✓ Use Airflow for orchestration
  - ✓ Support backfilling

✓ **Docker Requirements**
  - ✓ docker-compose build works
  - ✓ docker-compose up provides Airflow link
  - ✓ DAG runs successfully

✓ **Code Quality**
  - ✓ Well-structured and modular
  - ✓ Comprehensive configuration
  - ✓ Error handling
  - ✓ Logging and monitoring

## Files to Submit

1. **Code Artifacts** (in zip file):
   - ✓ Airflow pipeline (dags/ml_pipeline_dag.py)
   - ✓ Dockerfile
   - ✓ docker-compose.yaml
   - ✓ requirements.txt
   - ✓ src/ folder with all scripts
   - ✓ config/ folder
   - ✓ data/ folder
   - ✓ datamart/ folder structure
   - ✓ readme.txt with GitHub repo link

2. **Presentation Deck** (separate PDF, max 10 slides):
   - Slide 1: Pipeline Overview
   - Slide 2: Architecture Diagram
   - Slide 3: Model Training Strategy
   - Slide 4: Best Model Selection Process
   - Slide 5: Temporal Leakage Prevention
   - Slide 6: Model Monitoring Metrics
   - Slide 7: PSI Drift Detection
   - Slide 8: Performance Visualization Results
   - Slide 9: Model Governance & Retraining SOP
   - Slide 10: Deployment Strategy & Future Work

## Notes for Presentation Deck (Task 2)

### Technical Design Decisions to Highlight:
1. **Multi-model approach**: Why we train 3 different models
2. **Automatic selection**: AUC-ROC based selection with JSON config
3. **Temporal validation**: MOB=0 for features, MOB=6 for labels
4. **Monitoring strategy**: Performance metrics + PSI for drift
5. **Config-driven**: All models stored, best model selected via config
6. **Scalability**: PySpark for data processing, Airflow for orchestration

### Visualizations to Include:
1. Pipeline architecture diagram
2. Model performance comparison chart
3. PSI drift detection chart
4. Threshold compliance over time
5. Confusion matrix evolution

### Model Governance SOP:
1. **Monitoring Frequency**: Monthly batch evaluation
2. **Alert Triggers**: Performance below thresholds or PSI > 0.2
3. **Retraining Process**: Automated with rolling 12-month window
4. **Deployment**: Blue-green or canary deployment strategy
5. **Rollback Plan**: Keep previous model artifacts for quick rollback

---

## Completed! ✓

All components of the ML pipeline have been implemented:
- ✓ Model training with 3 algorithms
- ✓ Best model selection via config JSON
- ✓ Model inference using best model
- ✓ Comprehensive monitoring with metrics and PSI
- ✓ Visualization generation
- ✓ Airflow DAG orchestration
- ✓ Docker configuration
- ✓ Complete documentation

The pipeline is production-ready and follows ML engineering best practices!
