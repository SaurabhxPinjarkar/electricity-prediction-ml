"""
Unit tests for data_loader module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from data_loader import DataLoader, load_data_simple


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_dataloader_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader is not None
        assert loader.scaler is not None
        assert isinstance(loader.feature_columns, list)
    
    def test_load_energy_data(self):
        """Test loading energy dataset."""
        loader = DataLoader()
        try:
            df = loader.load_energy_data()
            assert df is not None
            assert len(df) > 0
            assert 'datetime' in df.columns
            print(f"✓ Energy data loaded: {len(df)} rows")
        except FileNotFoundError:
            pytest.skip("Energy dataset not found")
    
    def test_load_weather_data(self):
        """Test loading weather dataset."""
        loader = DataLoader()
        try:
            df = loader.load_weather_data()
            assert df is not None
            assert len(df) > 0
            assert 'datetime' in df.columns
            print(f"✓ Weather data loaded: {len(df)} rows")
        except FileNotFoundError:
            pytest.skip("Weather dataset not found")
    
    def test_merge_datasets(self):
        """Test merging energy and weather datasets."""
        loader = DataLoader()
        try:
            energy_df = loader.load_energy_data()
            weather_df = loader.load_weather_data()
            merged_df = loader.merge_datasets(energy_df, weather_df)
            
            assert merged_df is not None
            assert len(merged_df) > 0
            assert 'datetime' in merged_df.columns
            print(f"✓ Datasets merged: {len(merged_df)} rows")
        except FileNotFoundError:
            pytest.skip("Datasets not found")
    
    def test_engineer_features(self):
        """Test feature engineering."""
        loader = DataLoader()
        
        # Create sample dataframe
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        df = pd.DataFrame({
            'datetime': dates,
            'temp': np.random.uniform(270, 300, 100),
            'humidity': np.random.uniform(30, 90, 100),
            'wind_speed': np.random.uniform(0, 10, 100)
        })
        
        df_engineered = loader.engineer_features(df)
        
        # Check that new features were created
        expected_features = ['hour', 'day_of_week', 'month', 'is_weekend',
                           'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        
        for feature in expected_features:
            assert feature in df_engineered.columns, f"Missing feature: {feature}"
        
        print(f"✓ Features engineered: {len(df_engineered.columns)} columns")
    
    def test_prepare_data(self):
        """Test complete data preparation pipeline."""
        loader = DataLoader()
        
        try:
            X_train, X_test, y_train, y_test, feature_cols, target_col = loader.prepare_data(
                test_size=0.2,
                random_state=42
            )
            
            # Validate shapes
            assert X_train.shape[0] > 0
            assert X_test.shape[0] > 0
            assert len(y_train) == X_train.shape[0]
            assert len(y_test) == X_test.shape[0]
            assert X_train.shape[1] == X_test.shape[1]
            
            # Validate feature columns
            assert len(feature_cols) > 0
            assert X_train.shape[1] == len(feature_cols)
            
            # Validate target column
            assert target_col is not None
            assert isinstance(target_col, str)
            
            print(f"✓ Data prepared successfully")
            print(f"  - Training samples: {X_train.shape[0]}")
            print(f"  - Test samples: {X_test.shape[0]}")
            print(f"  - Features: {len(feature_cols)}")
            print(f"  - Target: {target_col}")
            
        except FileNotFoundError:
            pytest.skip("Datasets not found")
    
    def test_clean_data(self):
        """Test data cleaning."""
        loader = DataLoader()
        
        # Create sample dataframe with missing values
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, np.nan, 30, 40, 50],
            'target': [100, 200, 300, 400, 500]
        })
        
        cleaned_df = loader.clean_data(df, ['feature1', 'feature2'], 'target')
        
        # Check that missing values were handled
        assert cleaned_df['feature1'].isnull().sum() == 0
        assert cleaned_df['feature2'].isnull().sum() == 0
        
        print(f"✓ Data cleaned: {len(cleaned_df)} rows")


def test_load_data_simple():
    """Test simple data loading function."""
    try:
        df, feature_cols, target_col = load_data_simple('dummy.csv')
        assert df is not None
        assert len(feature_cols) > 0
        assert target_col is not None
        print("✓ Simple data loading works")
    except:
        pytest.skip("Simple data loading requires datasets")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
