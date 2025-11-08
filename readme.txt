https://github.com/vishal2505/SMU_MLE_Assignment_2

Project: End-to-End ML Pipeline for Loan Default Prediction
Assignment: MLE Assignment 2 - Machine Learning Pipelines

Implementation includes:
- Model Training: Logistic Regression, Random Forest, XGBoost
- Model Selection: Automatic best model selection via config JSON
- Model Inference: Predictions using best model
- Model Monitoring: Performance metrics + PSI drift detection
- Visualization: Performance and stability charts
- Orchestration: Apache Airflow DAG with monthly schedule

Quick Start:
1. docker-compose build
2. docker-compose up
3. Access Airflow at http://localhost:8080 (airflow/airflow)

See QUICKSTART.md for detailed instructions.
See IMPLEMENTATION_SUMMARY.md for complete feature list.
See README_PIPELINE.md for comprehensive documentation.