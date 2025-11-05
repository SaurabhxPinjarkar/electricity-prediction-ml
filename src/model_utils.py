"""
Model Utilities Module

Helper functions for model operations:
- Saving and loading models
- Saving and loading metrics
- Feature engineering helpers
- Preprocessing utilities
"""

import joblib
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd


def save_model(model_data: Dict[str, Any], filepath: str) -> None:
    """
    Save model and associated artifacts to disk.
    
    Args:
        model_data: Dictionary containing model, scaler, and metadata
        filepath: Path to save the model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model_data, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Dict[str, Any]:
    """
    Load model and associated artifacts from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Dictionary containing model, scaler, and metadata
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model_data = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    
    return model_data


def save_metrics(metrics: Dict[str, Any], filepath: str) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save metrics
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Metrics saved to {filepath}")


def load_metrics(filepath: str) -> Dict[str, Any]:
    """
    Load metrics from JSON file.
    
    Args:
        filepath: Path to metrics file
        
    Returns:
        Dictionary of metrics
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Metrics file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def extract_time_features(datetime_series: pd.Series) -> pd.DataFrame:
    """
    Extract time-based features from datetime series.
    
    Args:
        datetime_series: Pandas Series with datetime values
        
    Returns:
        DataFrame with time features
    """
    features = pd.DataFrame()
    
    features['hour'] = datetime_series.dt.hour
    features['day_of_week'] = datetime_series.dt.dayofweek
    features['month'] = datetime_series.dt.month
    features['day_of_year'] = datetime_series.dt.dayofyear
    features['week_of_year'] = datetime_series.dt.isocalendar().week
    features['is_weekend'] = (datetime_series.dt.dayofweek >= 5).astype(int)
    
    # Cyclical encoding
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
    features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    
    return features


def prepare_input_features(
    hour: int,
    temp_celsius: float,
    humidity: float,
    wind_speed: float = 2.0,
    pressure: float = 1013.0,
    is_weekend: bool = False,
    month: int = 1,
    day_of_week: int = 0,
    feature_columns: List[str] = None,
    scaler: Any = None,
    lag_values: Dict[str, float] = None
) -> np.ndarray:
    """
    Prepare input features for prediction.
    
    Args:
        hour: Hour of day (0-23)
        temp_celsius: Temperature in Celsius
        humidity: Humidity percentage
        wind_speed: Wind speed in m/s
        pressure: Atmospheric pressure in hPa
        is_weekend: Whether it's weekend
        month: Month (1-12)
        day_of_week: Day of week (0-6)
        feature_columns: List of expected feature columns
        scaler: Fitted scaler object
        lag_values: Dictionary with lag feature values
        
    Returns:
        Scaled feature array ready for prediction
    """
    # Create feature dictionary
    features = {
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': int(is_weekend),
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'day_sin': np.sin(2 * np.pi * day_of_week / 7),
        'day_cos': np.cos(2 * np.pi * day_of_week / 7),
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12),
        'temp_celsius': temp_celsius,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'pressure': pressure,
    }
    
    # Add lag features if provided
    if lag_values:
        features.update(lag_values)
    
    # Create DataFrame with proper column order
    if feature_columns:
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in features:
                # Default values for missing features
                if 'lag' in col or 'rolling' in col:
                    features[col] = 25000.0  # Approximate average consumption
                elif 'clouds' in col:
                    features[col] = 50.0
                elif 'rain' in col or 'is_' in col:
                    features[col] = 0.0
                elif 'price' in col:
                    features[col] = 50.0
                else:
                    features[col] = 0.0
        
        # Create array in correct order
        feature_array = np.array([[features[col] for col in feature_columns]])
    else:
        feature_array = np.array([[v for v in features.values()]])
    
    # Scale features if scaler provided
    if scaler:
        feature_array = scaler.transform(feature_array)
    
    return feature_array


def predict_single(model_data: Dict[str, Any], **kwargs) -> float:
    """
    Make a single prediction.
    
    Args:
        model_data: Dictionary containing model and artifacts
        **kwargs: Feature values for prediction
        
    Returns:
        Predicted consumption value
    """
    model = model_data['model']
    scaler = model_data['scaler']
    feature_columns = model_data['feature_columns']
    
    # Prepare features
    X = prepare_input_features(
        feature_columns=feature_columns,
        scaler=scaler,
        **kwargs
    )
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    return float(prediction)


def get_model_info(model_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract information about the model.
    
    Args:
        model_data: Dictionary containing model and artifacts
        
    Returns:
        Dictionary with model information
    """
    model = model_data['model']
    
    info = {
        'model_type': type(model).__name__,
        'n_features': len(model_data.get('feature_columns', [])),
        'feature_columns': model_data.get('feature_columns', []),
        'target_column': model_data.get('target_column', 'unknown'),
    }
    
    # Add model-specific parameters
    if hasattr(model, 'n_estimators'):
        info['n_estimators'] = model.n_estimators
    if hasattr(model, 'max_depth'):
        info['max_depth'] = model.max_depth
    if hasattr(model, 'feature_importances_'):
        info['has_feature_importance'] = True
    
    return info


def validate_input_ranges(
    hour: int = None,
    temp_celsius: float = None,
    humidity: float = None,
    wind_speed: float = None
) -> bool:
    """
    Validate that input values are in reasonable ranges.
    
    Args:
        hour: Hour value to validate
        temp_celsius: Temperature to validate
        humidity: Humidity to validate
        wind_speed: Wind speed to validate
        
    Returns:
        True if all values are valid
        
    Raises:
        ValueError: If any value is out of range
    """
    if hour is not None and not (0 <= hour <= 23):
        raise ValueError(f"Hour must be between 0 and 23, got {hour}")
    
    if temp_celsius is not None and not (-50 <= temp_celsius <= 60):
        raise ValueError(f"Temperature must be between -50°C and 60°C, got {temp_celsius}")
    
    if humidity is not None and not (0 <= humidity <= 100):
        raise ValueError(f"Humidity must be between 0 and 100, got {humidity}")
    
    if wind_speed is not None and not (0 <= wind_speed <= 50):
        raise ValueError(f"Wind speed must be between 0 and 50 m/s, got {wind_speed}")
    
    return True


if __name__ == "__main__":
    # Test utility functions
    print("Model utilities module loaded successfully!")
    
    # Test time feature extraction
    test_dates = pd.date_range('2024-01-01', periods=5, freq='H')
    time_features = extract_time_features(test_dates)
    print("\nTime features extracted:")
    print(time_features.head())
    
    # Test input validation
    try:
        validate_input_ranges(hour=12, temp_celsius=20, humidity=60, wind_speed=5)
        print("\n✓ Input validation passed!")
    except ValueError as e:
        print(f"\n✗ Input validation failed: {e}")
