"""
Model Training Script for Electricity Consumption Prediction

This script:
- Loads and prepares data using data_loader
- Trains a RandomForest model with hyperparameter tuning
- Evaluates model performance
- Saves trained model and metrics
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from model_utils import save_model, save_metrics


def get_hyperparameter_grid():
    """
    Define hyperparameter search space for RandomForest.
    
    Returns:
        Dictionary of hyperparameters to search
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }
    return param_grid


def train_model(X_train, y_train, hyperparameter_tuning=True, n_iter=10, cv=3, verbose=1):
    """
    Train RandomForest model with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        hyperparameter_tuning: Whether to perform hyperparameter search
        n_iter: Number of iterations for RandomizedSearchCV
        cv: Number of cross-validation folds
        verbose: Verbosity level
        
    Returns:
        Trained model
    """
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*60)
    
    if hyperparameter_tuning:
        print(f"\nPerforming hyperparameter tuning with {n_iter} iterations...")
        
        # Base model
        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Hyperparameter search
        param_grid = get_hyperparameter_grid()
        
        rf_search = RandomizedSearchCV(
            estimator=rf_base,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            verbose=verbose,
            random_state=42,
            n_jobs=-1,
            scoring='neg_mean_squared_error'
        )
        
        start_time = time.time()
        rf_search.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"\nBest parameters found: {rf_search.best_params_}")
        print(f"Best CV score (neg MSE): {rf_search.best_score_:.2f}")
        print(f"Training time: {training_time:.2f} seconds")
        
        model = rf_search.best_estimator_
    else:
        print("\nTraining model with default parameters...")
        
        model = RandomForestRegressor(
    n_estimators=50,
    max_depth=15,
    ...
    n_jobs=2,
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Training time: {training_time:.2f} seconds")
    
    return model, training_time


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Evaluate model performance on train and test sets.
    
    Args:
        model: Trained model
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        
    Returns:
        Dictionary of metrics
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for train set
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Calculate metrics for test set
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    metrics = {
        'train': {
            'rmse': float(train_rmse),
            'mae': float(train_mae),
            'r2': float(train_r2),
            'mape': float(train_mape)
        },
        'test': {
            'rmse': float(test_rmse),
            'mae': float(test_mae),
            'r2': float(test_r2),
            'mape': float(test_mape)
        }
    }
    
    # Print metrics
    print("\nTraining Set Metrics:")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  MAE:  {train_mae:.2f}")
    print(f"  R²:   {train_r2:.4f}")
    print(f"  MAPE: {train_mape:.2f}%")
    
    print("\nTest Set Metrics:")
    print(f"  RMSE: {test_rmse:.2f}")
    print(f"  MAE:  {test_mae:.2f}")
    print(f"  R²:   {test_r2:.4f}")
    print(f"  MAPE: {test_mape:.2f}%")
    
    # Check for overfitting
    if train_r2 - test_r2 > 0.1:
        print("\n⚠️  Warning: Possible overfitting detected (train R² >> test R²)")
    else:
        print("\n✓ Model generalization looks good!")
    
    return metrics


def get_feature_importance(model, feature_names, top_n=15):
    """
    Get and display feature importance.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to display
        
    Returns:
        DataFrame with feature importance
    """
    print("\n" + "="*60)
    print(f"TOP {top_n} FEATURE IMPORTANCE")
    print("="*60)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(importance_df.head(top_n).to_string(index=False))
    
    return importance_df


def main(data_path=None, output_path='model.pkl', hyperparameter_tuning=True, n_iter=10):
    """
    Main training pipeline.
    
    Args:
        data_path: Path to dataset (if None, uses default energy+weather datasets)
        output_path: Path to save trained model
        hyperparameter_tuning: Whether to perform hyperparameter tuning
        n_iter: Number of iterations for hyperparameter search
    """
    print("\n" + "="*60)
    print("ELECTRICITY CONSUMPTION PREDICTION - MODEL TRAINING")
    print("="*60)
    
    # Load and prepare data
    print("\n[1/5] Loading and preparing data...")
    loader = DataLoader()
    X_train, X_test, y_train, y_test, feature_cols, target_col = loader.prepare_data()
    
    # Train model
    print("\n[2/5] Training model...")
    model, training_time = train_model(
        X_train, y_train, 
        hyperparameter_tuning=hyperparameter_tuning,
        n_iter=n_iter
    )
    
    # Evaluate model
    print("\n[3/5] Evaluating model...")
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    metrics['training_time_seconds'] = training_time
    metrics['target_column'] = target_col
    metrics['n_features'] = len(feature_cols)
    metrics['n_train_samples'] = len(X_train)
    metrics['n_test_samples'] = len(X_test)
    
    # Feature importance
    print("\n[4/5] Analyzing feature importance...")
    importance_df = get_feature_importance(model, feature_cols)
    
    # Save model and artifacts
    print("\n[5/5] Saving model and metrics...")
    
    # Save model with scaler and feature info
    model_data = {
        'model': model,
        'scaler': loader.scaler,
        'feature_columns': feature_cols,
        'target_column': target_col,
        'feature_importance': importance_df.to_dict('records')
    }
    
    save_model(model_data, output_path)
    print(f"✓ Model saved to: {output_path}")
    
    # Save metrics
    metrics_path = Path(output_path).parent / 'metrics.json'
    save_metrics(metrics, metrics_path)
    print(f"✓ Metrics saved to: {metrics_path}")
    
    # Save feature importance
    importance_path = Path(output_path).parent / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved to: {importance_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\n🎯 Test R² Score: {metrics['test']['r2']:.4f}")
    print(f"📊 Test RMSE: {metrics['test']['rmse']:.2f}")
    print(f"📈 Test MAE: {metrics['test']['mae']:.2f}")
    print(f"⏱️  Training Time: {training_time:.2f} seconds")
    
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train electricity consumption prediction model')
    parser.add_argument('--data', type=str, default=None, help='Path to dataset CSV file')
    parser.add_argument('--out', type=str, default='model.pkl', help='Path to save trained model')
    parser.add_argument('--no-tuning', action='store_true', help='Skip hyperparameter tuning')
    parser.add_argument('--n-iter', type=int, default=10, help='Number of hyperparameter search iterations')
    
    args = parser.parse_args()
    
    model, metrics = main(
        data_path=args.data,
        output_path=args.out,
        hyperparameter_tuning=not args.no_tuning,
        n_iter=args.n_iter
    )

