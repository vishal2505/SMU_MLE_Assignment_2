# Quick Start Guide

## Prerequisites
- Docker Desktop installed and running
- At least 8GB RAM available
- Git (optional, for version control)

## Step-by-Step Setup

### 1. Build the Docker Images
```bash
cd /Users/vishalmishra/MyDocuments/SMU_MITB/Term-4/MLE/Assignment/mle_assignment_2
docker-compose build
```

This will:
- Build the Airflow image with Python 3.12
- Install Java for PySpark
- Install all Python dependencies from requirements.txt

**Expected time**: 5-10 minutes

### 2. Start the Services
```bash
docker-compose up
```

This will:
- Initialize the Airflow database
- Create the admin user (username: airflow, password: airflow)
- Start the Airflow webserver on port 8080
- Start the Airflow scheduler

**Wait for**: "Airflow is ready" message in logs

### 3. Access Airflow UI
Open your browser and navigate to:
```
http://localhost:8080
```

Login credentials:
- **Username**: airflow
- **Password**: airflow

### 4. Enable and Run the DAG

1. In the Airflow UI, you should see the `ml_pipeline` DAG
2. Click on the toggle switch to enable it
3. The DAG is scheduled to run monthly, but you can:
   - **Manual trigger**: Click on the "Play" button
   - **Backfill**: The DAG will automatically backfill from 2023-01-01

### 5. Monitor Execution

- Click on the DAG name to see the task graph
- Click on individual tasks to see logs
- Green = Success, Red = Failed, Yellow = Running

## DAG Execution Timeline

The pipeline processes data from **2023-01-01 to 2024-12-31**:

### Phase 1: Data Processing (2023-01-01 onwards)
- Bronze, Silver, Gold layers are created
- Feature store and Label store are built

### Phase 2: Model Training (2024-07-01 onwards)
- Requires 18 months of data (12 for training + 6 for labels)
- First training happens at 2024-07-01
- Models are saved to `model_store/`

### Phase 3: Inference (After training)
- Starts making predictions once models are trained
- Predictions saved to `datamart/gold/predictions/`

### Phase 4: Monitoring (6 months after inference)
- Requires both predictions and labels
- Evaluates model performance
- Calculates PSI for drift detection
- Saves results to `datamart/gold/monitoring/`

### Phase 5: Visualization (After monitoring)
- Generates performance charts
- Creates drift analysis plots
- Saves to `results/monitoring_visualizations/`

## Expected Outputs

### 1. Model Store
```
model_store/
├── model_config.json              # Best model: XGBoost/RandomForest/LogisticRegression
├── model_evaluation.json          # All models evaluation summary
├── model_monitoring.json          # Monitoring history
├── LogisticRegression/
├── RandomForest/
└── XGBoost/
```

### 2. Predictions
```
datamart/gold/predictions/
├── predictions_2024_07_01.parquet
├── predictions_2024_08_01.parquet
└── ...
```

### 3. Monitoring Results
```
datamart/gold/monitoring/
├── monitoring_2024_07_01.json
├── monitoring_2024_08_01.json
└── ...
```

### 4. Visualizations
```
results/monitoring_visualizations/
├── performance_metrics_over_time.png
├── psi_over_time.png
├── confusion_matrix_trend.png
├── threshold_compliance.png
├── prediction_distribution.png
└── monitoring_summary_report.txt
```

## Troubleshooting

### Issue: Docker build fails
**Solution**: Ensure Docker Desktop is running and has sufficient resources
```bash
# Check Docker status
docker --version
docker ps

# Increase Docker memory to 8GB in Docker Desktop settings
```

### Issue: Port 8080 already in use
**Solution**: Stop other services using port 8080 or change the port in docker-compose.yaml
```bash
# Find what's using port 8080
lsof -i :8080

# Kill the process or change docker-compose.yaml
ports:
  - "8081:8080"  # Changed to 8081
```

### Issue: DAG not appearing in UI
**Solution**: Check the logs for DAG parsing errors
```bash
docker-compose logs airflow-scheduler | grep ml_pipeline
```

### Issue: Tasks failing with "Module not found"
**Solution**: Rebuild the Docker image
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Issue: Out of memory errors
**Solution**: Reduce Spark memory in config/config.py
```python
SPARK_DRIVER_MEMORY = "2g"  # Reduced from 4g
```

## Stopping the Services

```bash
# Stop services (preserves data)
docker-compose down

# Stop and remove all data
docker-compose down -v
```

## Testing Individual Components

You can test individual scripts outside of Airflow:

### Test Model Training
```bash
docker exec -it <container_id> bash
cd /opt/airflow
python src/model_training.py --start_date 2023-01-01 --end_date 2024-01-01
```

### Test Inference
```bash
python src/model_inference.py --snapshot_date 2024-01-01
```

### Test Monitoring
```bash
python src/model_monitoring.py --snapshot_date 2024-07-01
```

### Test Visualization
```bash
python src/visualization.py
```

## Next Steps

1. **Review the DAG execution**: Monitor task progress in Airflow UI
2. **Check model performance**: Review `model_store/model_config.json`
3. **Analyze monitoring results**: Check `results/monitoring_visualizations/`
4. **Create presentation deck**: Use the visualizations and insights

## Important Notes

- The first complete end-to-end run will take **several hours** due to backfilling
- Model training only starts after sufficient data (18 months)
- Monitoring requires both predictions and labels (6-month delay)
- Visualizations are generated after monitoring completes

## Support

For issues or questions:
1. Check the logs: `docker-compose logs -f`
2. Review the README_PIPELINE.md for detailed documentation
3. Check IMPLEMENTATION_SUMMARY.md for component details

## Success Criteria Checklist

- [ ] Docker build completes without errors
- [ ] Docker compose up starts Airflow successfully
- [ ] Airflow UI accessible at http://localhost:8080
- [ ] `ml_pipeline` DAG visible in UI
- [ ] DAG tasks execute successfully (green status)
- [ ] Model artifacts created in `model_store/`
- [ ] `model_config.json` identifies best model
- [ ] Predictions saved to gold layer
- [ ] Monitoring results generated
- [ ] Visualizations created in results folder

Once all tasks are green, the pipeline is working correctly! ✓
