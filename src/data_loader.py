"""
Data Loader Module for Electricity Consumption Prediction

This module handles:
- Loading energy and weather datasets
- Merging data on timestamp
- Feature engineering (extracting time features, weather features)
- Data cleaning and preprocessing
- Train/test splitting
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Main data loading and preprocessing class for electricity consumption prediction.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing the datasets. If None, uses parent dataset directory.
        """
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent.parent
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.target_column = None
        
    def load_energy_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load the energy dataset.
        
        Args:
            filepath: Path to energy dataset CSV file
            
        Returns:
            DataFrame with energy data
        """
        if filepath is None:
            # Try to find energy dataset in common locations
            possible_paths = [
                Path(__file__).parent.parent / 'data' / 'energy_dataset.csv',  # Streamlit Cloud path
                self.data_dir / 'data' / 'energy_dataset.csv',
                self.data_dir / 'dataset' / 'LATEST_DATASET_ENERGY' / 'energy_dataset.csv',
                self.data_dir / 'energy_dataset.csv',
                self.data_dir / 'dataset.csv',
            ]
            
            # Add Windows local path only if on Windows
            if os.name == 'nt':
                possible_paths.append(Path(r's:\Saurabh Pinjarkar\dataset\LATEST_DATASET_ENERGY\energy_dataset.csv'))
            
            # Remove the trailing empty list item
            possible_paths = [p for p in possible_paths if p]
            
            for path in possible_paths:
                if Path(path).exists():
                    filepath = path
                    break
            
            if filepath is None:
                raise FileNotFoundError("Energy dataset not found. Please specify filepath.")
        
        print(f"Loading energy data from: {filepath}")
        df = pd.read_csv(filepath)
        
        # Parse datetime - ensure it's datetime type after renaming
        if 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'], utc=True)
            df['datetime'] = df['datetime'].dt.tz_localize(None)  # Remove timezone info
            df = df.drop(columns=['time'])
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            df['datetime'] = df['datetime'].dt.tz_localize(None)  # Remove timezone info
        
        return df
    
    def load_weather_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load the weather features dataset.
        
        Args:
            filepath: Path to weather dataset CSV file
            
        Returns:
            DataFrame with weather data
        """
        if filepath is None:
            # Try to find weather dataset in common locations
            possible_paths = [
                Path(__file__).parent.parent / 'data' / 'weather_features.csv',  # Streamlit Cloud path
                self.data_dir / 'data' / 'weather_features.csv',
                self.data_dir / 'dataset' / 'LATEST_DATASET_ENERGY' / 'weather_features.csv',
                self.data_dir / 'weather_features.csv',
            ]
            
            # Add Windows local path only if on Windows
            if os.name == 'nt':
                possible_paths.append(Path(r's:\Saurabh Pinjarkar\dataset\LATEST_DATASET_ENERGY\weather_features.csv'))
            
            for path in possible_paths:
                if Path(path).exists():
                    filepath = path
                    break
            
            if filepath is None:
                raise FileNotFoundError("Weather dataset not found. Please specify filepath.")
        
        print(f"Loading weather data from: {filepath}")
        df = pd.read_csv(filepath)
        
        # Parse datetime - ensure it's datetime type after renaming
        if 'dt_iso' in df.columns:
            df['datetime'] = pd.to_datetime(df['dt_iso'], utc=True)
            df['datetime'] = df['datetime'].dt.tz_localize(None)  # Remove timezone info
            df = df.drop(columns=['dt_iso'])
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            df['datetime'] = df['datetime'].dt.tz_localize(None)  # Remove timezone info
        
        return df
    
    def merge_datasets(self, energy_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge energy and weather datasets on datetime.
        
        Args:
            energy_df: Energy consumption DataFrame
            weather_df: Weather features DataFrame
            
        Returns:
            Merged DataFrame
        """
        print("Merging energy and weather data...")
        
        # Ensure both have datetime column
        if 'datetime' not in energy_df.columns or 'datetime' not in weather_df.columns:
            raise ValueError("Both dataframes must have a 'datetime' column")
        
        # Merge on datetime
        merged_df = pd.merge(energy_df, weather_df, on='datetime', how='inner')
        
        print(f"Merged dataset shape: {merged_df.shape}")
        return merged_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from existing data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        print("Engineering features...")
        df = df.copy()
        
        # Ensure datetime column is properly converted to datetime type
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Time-based features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        df['day_of_year'] = df['datetime'].dt.dayofyear
        df['week_of_year'] = df['datetime'].dt.isocalendar().week
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for hour (important for time patterns)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Temperature features (clean if needed)
        if 'temp' in df.columns:
            # Convert from Kelvin to Celsius if values are too high
            if df['temp'].mean() > 100:
                df['temp_celsius'] = df['temp'] - 273.15
            else:
                df['temp_celsius'] = df['temp']
            
            # Temperature bins
            df['temp_category'] = pd.cut(df['temp_celsius'], 
                                         bins=[-np.inf, 0, 10, 20, 30, np.inf],
                                         labels=[0, 1, 2, 3, 4])
        
        # Humidity features
        if 'humidity' in df.columns:
            df['humidity_category'] = pd.cut(df['humidity'], 
                                            bins=[0, 30, 60, 100],
                                            labels=[0, 1, 2])
        
        # Wind speed features
        if 'wind_speed' in df.columns:
            df['wind_category'] = pd.cut(df['wind_speed'],
                                        bins=[0, 2, 5, 10, np.inf],
                                        labels=[0, 1, 2, 3])
        
        # Cloud cover features
        if 'clouds_all' in df.columns:
            df['is_cloudy'] = (df['clouds_all'] > 50).astype(int)
        
        # Rain features
        if 'rain_1h' in df.columns:
            df['is_raining'] = (df['rain_1h'] > 0).astype(int)
        
        return df
    
    def select_target(self, df: pd.DataFrame) -> str:
        """
        Automatically detect and select the target column.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Name of target column
        """
        # Possible target column names
        target_candidates = [
            'total load actual',
            'total_load_actual',
            'Electricity_Consumption',
            'consumption',
            'load',
            'demand'
        ]
        
        for candidate in target_candidates:
            if candidate in df.columns:
                print(f"Selected target column: {candidate}")
                return candidate
        
        # If no match, look for columns with 'load' or 'consumption'
        for col in df.columns:
            if 'load' in col.lower() or 'consumption' in col.lower():
                print(f"Selected target column: {col}")
                return col
        
        raise ValueError("Could not automatically detect target column. Please specify manually.")
    
    def select_features(self, df: pd.DataFrame, target_col: str) -> list:
        """
        Select relevant features for modeling.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            List of feature column names
        """
        # Engineered time features
        time_features = [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
        ]
        
        # Weather features
        weather_features = []
        if 'temp_celsius' in df.columns:
            weather_features.extend(['temp_celsius'])
        if 'humidity' in df.columns:
            weather_features.append('humidity')
        if 'wind_speed' in df.columns:
            weather_features.append('wind_speed')
        if 'pressure' in df.columns:
            weather_features.append('pressure')
        if 'clouds_all' in df.columns:
            weather_features.append('clouds_all')
        if 'is_raining' in df.columns:
            weather_features.append('is_raining')
        
        # Price features (if available)
        price_features = []
        if 'price day ahead' in df.columns:
            price_features.append('price day ahead')
        if 'price actual' in df.columns:
            price_features.append('price actual')
        
        # Lag features for consumption (use previous hour)
        if target_col in df.columns:
            df[f'{target_col}_lag1'] = df[target_col].shift(1)
            df[f'{target_col}_lag24'] = df[target_col].shift(24)  # same hour yesterday
            df[f'{target_col}_rolling_mean_24'] = df[target_col].rolling(window=24, min_periods=1).mean()
        
        lag_features = [f'{target_col}_lag1', f'{target_col}_lag24', f'{target_col}_rolling_mean_24']
        
        # Combine all features
        all_features = time_features + weather_features + price_features + lag_features
        
        # Filter to only existing columns
        available_features = [f for f in all_features if f in df.columns]
        
        print(f"Selected {len(available_features)} features: {available_features}")
        return available_features
    
    def clean_data(self, df: pd.DataFrame, feature_cols: list, target_col: str) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature columns
            target_col: Target column name
            
        Returns:
            Cleaned DataFrame
        """
        print("Cleaning data...")
        df = df.copy()
        
        # Drop rows with missing target
        df = df.dropna(subset=[target_col])
        
        # Handle missing values in features
        for col in feature_cols:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['float64', 'int64']:
                    # Fill with median for numeric columns
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    # Fill with mode for categorical
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Remove outliers using IQR method for target
        Q1 = df[target_col].quantile(0.01)
        Q3 = df[target_col].quantile(0.99)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]
        
        print(f"Dataset shape after cleaning: {df.shape}")
        return df
    
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, str]:
        """
        Complete data preparation pipeline.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test, feature_columns, target_column
        """
        # Load datasets
        energy_df = self.load_energy_data()
        weather_df = self.load_weather_data()
        
        # Merge datasets
        df = self.merge_datasets(energy_df, weather_df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Select target
        target_col = self.select_target(df)
        self.target_column = target_col
        
        # Select features
        feature_cols = self.select_features(df, target_col)
        self.feature_columns = feature_cols
        
        # Clean data
        df = self.clean_data(df, feature_cols, target_col)
        
        # Extract X and y
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nData preparation complete!")
        print(f"Training set size: {X_train_scaled.shape}")
        print(f"Test set size: {X_test_scaled.shape}")
        print(f"Target column: {target_col}")
        print(f"Number of features: {len(feature_cols)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, target_col


def load_data_simple(filepath: str) -> Tuple[pd.DataFrame, list, str]:
    """
    Simple function to load a single CSV file for the Streamlit app.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame, feature columns, target column
    """
    loader = DataLoader()
    
    # Try to load as energy + weather datasets
    try:
        X_train, X_test, y_train, y_test, feature_cols, target_col = loader.prepare_data()
        
        # Combine back for returning full dataset
        X_full = np.vstack([X_train, X_test])
        y_full = np.concatenate([y_train, y_test])
        
        df = pd.DataFrame(X_full, columns=feature_cols)
        df[target_col] = y_full
        
        return df, feature_cols, target_col
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    X_train, X_test, y_train, y_test, feature_cols, target_col = loader.prepare_data()
    
    print("\n" + "="*50)
    print("Data Loader Test Complete!")
    print("="*50)
