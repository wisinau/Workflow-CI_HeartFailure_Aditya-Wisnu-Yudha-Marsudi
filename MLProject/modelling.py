"""
KRITERIA 3: Model Training for CI/CD
Automatically run by GitHub Actions
"""

import os
import sys
import pandas as pd
import numpy as np

# Setup paths
sys.path.insert(0, os.path.dirname(__file__))

print("="*70)
print("HEART FAILURE PREDICTION - CI/CD AUTOMATED TRAINING")
print("="*70)

# Step 1: Load data
print("\n📥 STEP 1: Loading preprocessed data...")
try:
    data_path = 'data_preprocessing'
    X_train = pd.read_csv(os.path.join(data_path, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_path, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_path, 'y_train.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(data_path, 'y_test.csv')).squeeze()
    
    print(f"   ✓ Data loaded successfully")
    print(f"   ✓ Train set: {X_train.shape}")
    print(f"   ✓ Test set: {X_test.shape}")
    
except FileNotFoundError as e:
    print(f"   ❌ Error: {e}")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Contents: {os.listdir('.')}")
    if os.path.exists('data_preprocessing'):
        print(f"   data_preprocessing contents: {os.listdir('data_preprocessing')}")
    exit(1)

# Step 2: Import ML libraries
print("\n📚 STEP 2: Importing libraries...")
try:
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, 
        roc_auc_score, confusion_matrix, classification_report
    )
    print("   ✓ All libraries imported successfully")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    exit(1)

# Step 3: Setup MLflow
print("\n⚙️  STEP 3: Setting up MLflow...")
mlflow.set_experiment("HeartFailure_CI_Pipeline")
print("   ✓ Experiment: HeartFailure_CI_Pipeline")

# Step 4: Train model
print("\n🤖 STEP 4: Training RandomForest model...")
try:
    with mlflow.start_run(run_name="ci_random_forest"):
        mlflow.sklearn.autolog()
        
        # Model parameters
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42
        }
        
        print(f"   Parameters: {params}")
        
        # Train
        model = RandomForestClassifier(**params, n_jobs=-1)
        model.fit(X_train, y_train)
        print("   ✓ Model trained successfully")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print("\n📊 STEP 5: Model Performance Metrics")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")

except Exception as e:
    print(f"   ❌ Training error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*70)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nArtifacts saved to MLflow")
print("Run 'mlflow ui' to view results")
