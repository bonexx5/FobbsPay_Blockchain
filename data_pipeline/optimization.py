import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import joblib
import json
import gzip
import os
from pathlib import Path

class DataOptimizer:
    """
    Data optimization pipeline for FobbsPay blockchain transactions
    Handles data cleaning, transformation, feature engineering, and optimization
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.scaler = None
        self.anomaly_detector = None
        self.clustering_model = None
        
    @staticmethod
    def _load_config(config_path: Optional[str]) -> Dict:
        """Load configuration from JSON file"""
        default_config = {
            "data_cleaning": {
                "drop_na_threshold": 0.7,
                "numeric_fill_strategy": "median",
                "categorical_fill_strategy": "mode"
            },
            "feature_engineering": {
                "window_sizes": [3, 7, 30],
                "time_features": True,
                "interaction_terms": True
            },
            "optimization": {
                "compression": True,
                "compression_level": 3,
                "categorical_encoding": "onehot",
                "normalization": "standard",
                "anomaly_detection": True,
                "clustering": True,
                "n_clusters": 5
            },
            "output": {
                "format": "parquet",
                "partition_columns": ["date"],
                "chunk_size": 100000
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return {**default_config, **json.load(f)}
        return default_config
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw transaction data
        - Handle missing values
        - Remove duplicates
        - Fix data types
        - Standardize formats
        """
        # Drop columns with too many missing values
        drop_threshold = self.config["data_cleaning"]["drop_na_threshold"]
        df = df.loc[:, df.isna().mean() < drop_threshold]
        
        # Fill missing values
        num_strategy = self.config["data_cleaning"]["numeric_fill_strategy"]
        cat_strategy = self.config["data_cleaning"]["categorical_fill_strategy"]
        
        for col in df.select_dtypes(include=np.number):
            if num_strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif num_strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif num_strategy == "zero":
                df[col] = df[col].fillna(0)
        
        for col in df.select_dtypes(exclude=np.number):
            if cat_strategy == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])
            elif cat_strategy == "missing":
                df[col] = df[col].fillna("missing")
        
        # Convert date columns
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Remove exact duplicates
        df = df.drop_duplicates()
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from raw data
        - Time-based features
        - Rolling statistics
        - Interaction terms
        - Aggregated features
        """
        config = self.config["feature_engineering"]
        
        # Time-based features
        if config["time_features"]:
            date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            for col in date_cols:
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_hour"] = df[col].dt.hour
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                df[f"{col}_is_weekend"] = df[col].dt.dayofweek >= 5
        
        # Rolling statistics for transaction amounts
        if 'amount' in df.columns and config["window_sizes"]:
            for window in config["window_sizes"]:
                df[f"amount_rolling_mean_{window}"] = df.groupby('sender')['amount'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                df[f"amount_rolling_std_{window}"] = df.groupby('sender')['amount'].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
        
        # Interaction terms
        if config["interaction_terms"] and 'sender' in df.columns and 'recipient' in df.columns:
            df['sender_recipient_pair'] = df['sender'] + "_" + df['recipient']
        
        return df
    
    def optimize_data(self, df: pd.DataFrame, fit_models: bool = True) -> pd.DataFrame:
        """
        Optimize data for storage and processing
        - Normalize/scale numeric features
        - Encode categorical variables
        - Detect anomalies
        - Cluster similar transactions
        - Compress data
        """
        config = self.config["optimization"]
        
        # Normalize numeric features
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols and config["normalization"]:
            if config["normalization"] == "standard":
                self.scaler = StandardScaler()
            elif config["normalization"] == "minmax":
                self.scaler = MinMaxScaler()
            
            if fit_models:
                df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
            else:
                df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        # Encode categorical variables
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        if categorical_cols and config["categorical_encoding"]:
            if config["categorical_encoding"] == "onehot":
                df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
            elif config["categorical_encoding"] == "label":
                for col in categorical_cols:
                    df[col] = df[col].astype('category').cat.codes
        
        # Anomaly detection
        if config["anomaly_detection"] and numeric_cols:
            if fit_models:
                self.anomaly_detector = IsolationForest(contamination=0.01, random_state=42)
                df['anomaly_score'] = self.anomaly_detector.fit_predict(df[numeric_cols])
            elif self.anomaly_detector:
                df['anomaly_score'] = self.anomaly_detector.predict(df[numeric_cols])
        
        # Transaction clustering
        if config["clustering"] and numeric_cols:
            if fit_models:
                self.clustering_model = KMeans(n_clusters=config["n_clusters"], random_state=42)
                df['cluster'] = self.clustering_model.fit_predict(df[numeric_cols])
            elif self.clustering_model:
                df['cluster'] = self.clustering_model.predict(df[numeric_cols])
        
        return df
    
    def process_data(self, input_path: str, output_path: str, fit_models: bool = True) -> None:
        """
        Complete data processing pipeline:
        1. Load raw data
        2. Clean data
        3. Engineer features
        4. Optimize data
        5. Save processed data
        """
        # Read input data
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.endswith('.parquet'):
            df = pd.read_parquet(input_path)
        elif input_path.endswith('.json'):
            df = pd.read_json(input_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Process data
        df = self.clean_data(df)
        df = self.engineer_features(df)
        df = self.optimize_data(df, fit_models=fit_models)
        
        # Save output
        output_format = self.config["output"]["format"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if output_format == "parquet":
            df.to_parquet(output_path, compression='gzip')
        elif output_format == "csv":
            df.to_csv(output_path, index=False)
        elif output_format == "json":
            df.to_json(output_path, orient='records')
        
        # Save models if fit
        if fit_models:
            model_dir = os.path.join(os.path.dirname(output_path), 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            if self.scaler:
                joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.joblib'))
            if self.anomaly_detector:
                joblib.dump(self.anomaly_detector, os.path.join(model_dir, 'anomaly_detector.joblib'))
            if self.clustering_model:
                joblib.dump(self.clustering_model, os.path.join(model_dir, 'clustering_model.joblib'))
    
    def compress_data(self, input_path: str, output_path: str) -> None:
        """
        Compress data files using gzip
        """
        compression_level = self.config["optimization"]["compression_level"]
        
        with open(input_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb', compresslevel=compression_level) as f_out:
                f_out.writelines(f_in)
    
    def load_models(self, model_dir: str) -> None:
        """Load trained models from directory"""
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        anomaly_path = os.path.join(model_dir, 'anomaly_detector.joblib')
        cluster_path = os.path.join(model_dir, 'clustering_model.joblib')
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        if os.path.exists(anomaly_path):
            self.anomaly_detector = joblib.load(anomaly_path)
        if os.path.exists(cluster_path):
            self.clustering_model = joblib.load(cluster_path)
