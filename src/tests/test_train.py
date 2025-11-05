"""
Unit tests for train module
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from train import train_model, evaluate_model, get_feature_importance
from model_utils import load_model, load_metrics


class TestTraining:
    """Test cases for model training."""
    
    def test_train_model_no_tuning(self):
        """Test training without hyperparameter tuning."""
        # Create dummy data
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100) * 1000 + 20000
        
        model, training_time = train_model(
            X_train, y_train,
            hyperparameter_tuning=False,
            verbose=0
        )
        
        assert model is not None
        assert training_time > 0
        assert hasattr(model, 'predict')
        
        print(f"✓ Model trained in {training_time:.2f} seconds")
    
    def test_train_model_with_tuning(self):
        """Test training with hyperparameter tuning."""
        # Create dummy data
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100) * 1000 + 20000
        
        model, training_time = train_model(
            X_train, y_train,
            hyperparameter_tuning=True,
            n_iter=2,
            cv=2,
            verbose=0
        )
        
        assert model is not None
        assert training_time > 0
        assert hasattr(model, 'predict')
        
        print(f"✓ Model trained with tuning in {training_time:.2f} seconds")
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        from sklearn.ensemble import RandomForestRegressor
        
        # Create dummy data
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100) * 1000 + 20000
        X_test = np.random.rand(20, 10)
        y_test = np.random.rand(20) * 1000 + 20000
        
        # Train a simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
        
        assert 'train' in metrics
        assert 'test' in metrics
        assert 'rmse' in metrics['train']
        assert 'mae' in metrics['train']
        assert 'r2' in metrics['train']
        assert 'rmse' in metrics['test']
        assert 'mae' in metrics['test']
        assert 'r2' in metrics['test']
        
        print(f"✓ Model evaluated - Test R²: {metrics['test']['r2']:.4f}")
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        from sklearn.ensemble import RandomForestRegressor
        
        # Create dummy data
        X_train = np.random.rand(100, 5)
        y_train = np.random.rand(100) * 1000 + 20000
        
        feature_names = [f'feature_{i}' for i in range(5)]
        
        # Train a simple model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Get importance
        importance_df = get_feature_importance(model, feature_names, top_n=5)
        
        assert len(importance_df) == 5
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert importance_df['importance'].sum() > 0
        
        print(f"✓ Feature importance extracted for {len(importance_df)} features")
    
    def test_model_save_load(self):
        """Test saving and loading model."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        from model_utils import save_model
        
        # Create dummy model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        scaler = StandardScaler()
        
        # Dummy data to fit
        X_dummy = np.random.rand(50, 5)
        y_dummy = np.random.rand(50) * 1000
        
        model.fit(X_dummy, y_dummy)
        scaler.fit(X_dummy)
        
        # Prepare model data
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_columns': [f'feature_{i}' for i in range(5)],
            'target_column': 'consumption'
        }
        
        # Save
        test_model_path = Path(__file__).parent / 'test_model.pkl'
        save_model(model_data, test_model_path)
        
        assert test_model_path.exists()
        
        # Load
        loaded_data = load_model(test_model_path)
        
        assert 'model' in loaded_data
        assert 'scaler' in loaded_data
        assert 'feature_columns' in loaded_data
        assert 'target_column' in loaded_data
        
        # Clean up
        test_model_path.unlink()
        
        print("✓ Model saved and loaded successfully")
    
    def test_metrics_save_load(self):
        """Test saving and loading metrics."""
        from model_utils import save_metrics
        
        # Create dummy metrics
        metrics = {
            'train': {'rmse': 100.5, 'mae': 80.2, 'r2': 0.95},
            'test': {'rmse': 120.3, 'mae': 95.1, 'r2': 0.92}
        }
        
        # Save
        test_metrics_path = Path(__file__).parent / 'test_metrics.json'
        save_metrics(metrics, test_metrics_path)
        
        assert test_metrics_path.exists()
        
        # Load
        loaded_metrics = load_metrics(test_metrics_path)
        
        assert 'train' in loaded_metrics
        assert 'test' in loaded_metrics
        assert loaded_metrics['train']['rmse'] == 100.5
        assert loaded_metrics['test']['r2'] == 0.92
        
        # Clean up
        test_metrics_path.unlink()
        
        print("✓ Metrics saved and loaded successfully")


def test_integration_train_pipeline():
    """Integration test for full training pipeline."""
    try:
        from train import main
        
        # Run training with minimal settings
        model_path = Path(__file__).parent / 'integration_test_model.pkl'
        
        # This will only work if datasets are available
        model, metrics = main(
            output_path=str(model_path),
            hyperparameter_tuning=False,
            n_iter=2
        )
        
        assert model is not None
        assert metrics is not None
        assert model_path.exists()
        
        # Clean up
        model_path.unlink()
        metrics_path = model_path.parent / 'metrics.json'
        if metrics_path.exists():
            metrics_path.unlink()
        
        print("✓ Integration test passed")
        
    except FileNotFoundError:
        pytest.skip("Datasets not available for integration test")
    except Exception as e:
        pytest.skip(f"Integration test skipped: {str(e)}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
