"""
OSHA Data Processing Module

This module handles cleaning, preprocessing, and feature engineering of OSHA data
for machine learning model training and prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from datetime import datetime, timedelta
import yaml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OSHADataProcessor:
    """Processes and cleans OSHA inspection and violation data."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the processor with configuration."""
        self.config = self._load_config(config_path)
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encoders and scalers
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'data_processing': {
                'missing_value_strategy': {
                    'numerical': 'median',
                    'categorical': 'mode'
                },
                'outlier_handling': {
                    'method': 'iqr',
                    'threshold': 1.5
                },
                'validation': {
                    'test_size': 0.2,
                    'validation_size': 0.2,
                    'random_state': 42,
                    'stratify': True
                }
            },
            'feature_engineering': {
                'temporal_features': [
                    'month_of_inspection',
                    'quarter_of_inspection',
                    'days_since_last_inspection'
                ],
                'industry_features': [
                    'naics_2_digit_risk_score',
                    'high_hazard_industry_flag'
                ],
                'violation_features': [
                    'violation_count_last_year',
                    'serious_violation_rate',
                    'repeat_violation_flag',
                    'willful_violation_flag'
                ]
            }
        }
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw OSHA inspection and violation data.
        
        Returns:
            Tuple of (inspections_df, violations_df)
        """
        try:
            # Combine data from multiple years
            inspections_files = list(self.raw_dir.glob("osha_inspections_*.csv"))
            violations_files = list(self.raw_dir.glob("osha_violations_*.csv"))
            
            if not inspections_files or not violations_files:
                raise FileNotFoundError("No OSHA data files found in raw directory")
            
            # Load and combine inspection data
            inspections_dfs = []
            for file in inspections_files:
                df = pd.read_csv(file)
                inspections_dfs.append(df)
            
            inspections_df = pd.concat(inspections_dfs, ignore_index=True)
            
            # Load and combine violation data
            violations_dfs = []
            for file in violations_files:
                df = pd.read_csv(file)
                violations_dfs.append(df)
            
            violations_df = pd.concat(violations_dfs, ignore_index=True)
            
            logger.info(f"Loaded {len(inspections_df)} inspections and {len(violations_df)} violations")
            
            return inspections_df, violations_df
            
        except Exception as e:
            logger.error(f"Error loading raw data: {str(e)}")
            raise
    
    def clean_data(self, inspections_df: pd.DataFrame, violations_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Clean and preprocess the raw data.
        
        Args:
            inspections_df: Raw inspections data
            violations_df: Raw violations data
            
        Returns:
            Tuple of cleaned (inspections_df, violations_df)
        """
        logger.info("Starting data cleaning...")
        
        # Clean inspections data
        inspections_clean = self._clean_inspections(inspections_df.copy())
        
        # Clean violations data
        violations_clean = self._clean_violations(violations_df.copy())
        
        logger.info("Data cleaning completed")
        
        return inspections_clean, violations_clean
    
    def _clean_inspections(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the inspections dataframe."""
        
        # Convert date columns
        df['open_date'] = pd.to_datetime(df['open_date'])
        
        # Handle missing values
        missing_strategy = self.config['data_processing']['missing_value_strategy']
        
        # Fill missing categorical values
        categorical_cols = ['inspection_type', 'scope', 'naics_code']
        for col in categorical_cols:
            if col in df.columns:
                if missing_strategy['categorical'] == 'mode':
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
                elif missing_strategy['categorical'] == 'constant':
                    df[col] = df[col].fillna('Unknown')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['inspection_id'])
        
        # Filter out invalid dates
        current_date = datetime.now()
        df = df[df['open_date'] <= current_date]
        
        # Add derived fields
        df['year'] = df['open_date'].dt.year
        df['month'] = df['open_date'].dt.month
        df['quarter'] = df['open_date'].dt.quarter
        df['day_of_week'] = df['open_date'].dt.dayofweek
        
        return df
    
    def _clean_violations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the violations dataframe."""
        
        # Handle missing penalty amounts
        df['penalty_amount'] = df['penalty_amount'].fillna(0)
        
        # Convert boolean columns
        boolean_cols = ['is_serious', 'is_repeat', 'is_willful']
        for col in boolean_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False)
        
        # Remove violations with invalid inspection IDs
        df = df.dropna(subset=['inspection_id'])
        
        # Handle outliers in penalty amounts
        if self.config['data_processing']['outlier_handling']['method'] == 'iqr':
            df = self._remove_outliers_iqr(df, 'penalty_amount')
        
        return df
    
    def _remove_outliers_iqr(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        threshold = self.config['data_processing']['outlier_handling']['threshold']
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        before_count = len(df)
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        after_count = len(df)
        
        logger.info(f"Removed {before_count - after_count} outliers from {column}")
        
        return df
    
    def engineer_features(self, inspections_df: pd.DataFrame, violations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for machine learning.
        
        Args:
            inspections_df: Cleaned inspections data
            violations_df: Cleaned violations data
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        
        # Start with inspections as base
        features_df = inspections_df.copy()
        
        # Add violation-based features
        violation_features = self._create_violation_features(violations_df)
        features_df = features_df.merge(violation_features, on='inspection_id', how='left')
        
        # Add temporal features
        features_df = self._add_temporal_features(features_df)
        
        # Add industry risk features
        features_df = self._add_industry_features(features_df)
        
        # Add establishment history features
        features_df = self._add_establishment_history_features(features_df, violations_df)
        
        logger.info(f"Feature engineering completed. Features shape: {features_df.shape}")
        
        return features_df
    
    def _create_violation_features(self, violations_df: pd.DataFrame) -> pd.DataFrame:
        """Create features from violations data."""
        
        # Aggregate violations by inspection
        violation_agg = violations_df.groupby('inspection_id').agg({
            'violation_id': 'count',
            'penalty_amount': ['sum', 'mean', 'max'],
            'is_serious': 'sum',
            'is_repeat': 'sum',
            'is_willful': 'sum',
            'standard_violated': 'nunique'
        }).round(2)
        
        # Flatten column names
        violation_agg.columns = [
            'total_violations',
            'total_penalty_amount',
            'avg_penalty_amount',
            'max_penalty_amount',
            'serious_violations_count',
            'repeat_violations_count',
            'willful_violations_count',
            'unique_standards_violated'
        ]
        
        # Calculate rates
        violation_agg['serious_violation_rate'] = (
            violation_agg['serious_violations_count'] / violation_agg['total_violations']
        ).fillna(0)
        
        violation_agg['repeat_violation_flag'] = (violation_agg['repeat_violations_count'] > 0).astype(int)
        violation_agg['willful_violation_flag'] = (violation_agg['willful_violations_count'] > 0).astype(int)
        
        # Create severity score
        violation_agg['violation_severity_score'] = (
            violation_agg['serious_violations_count'] * 3 +
            violation_agg['willful_violations_count'] * 5 +
            violation_agg['repeat_violations_count'] * 4
        )
        
        violation_agg = violation_agg.reset_index()
        
        return violation_agg
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features."""
        
        # Sort by establishment and date for lag features
        df = df.sort_values(['establishment_id', 'open_date'])
        
        # Days since last inspection for same establishment
        df['days_since_last_inspection'] = df.groupby('establishment_id')['open_date'].diff().dt.days
        df['days_since_last_inspection'] = df['days_since_last_inspection'].fillna(365)  # Default to 1 year for first inspection
        
        # Inspection frequency (inspections per year for establishment)
        df['inspection_year'] = df['open_date'].dt.year
        inspection_freq = df.groupby(['establishment_id', 'inspection_year']).size().reset_index(name='inspections_per_year')
        df = df.merge(inspection_freq, on=['establishment_id', 'inspection_year'], how='left')
        
        # Seasonal features
        df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        df['is_winter'] = ((df['month'] <= 2) | (df['month'] == 12)).astype(int)
        
        return df
    
    def _add_industry_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add industry-specific risk features."""
        
        # High-hazard industries (based on common NAICS codes with higher injury rates)
        high_hazard_naics = ['236220', '238210', '484121', '561720']  # Construction, transportation, etc.
        df['high_hazard_industry_flag'] = df['naics_code'].isin(high_hazard_naics).astype(int)
        
        # Industry risk score based on historical injury rates
        industry_risk_scores = {
            '236220': 0.85,  # Construction
            '238210': 0.78,  # Electrical contractors
            '484121': 0.72,  # General freight trucking
            '561720': 0.68,  # Janitorial services
            '311612': 0.45   # Meat processing
        }
        
        df['industry_risk_score'] = df['naics_code'].map(industry_risk_scores).fillna(0.5)
        
        # NAICS 2-digit grouping
        df['naics_2_digit'] = df['naics_code'].astype(str).str[:2]
        
        return df
    
    def _add_establishment_history_features(self, inspections_df: pd.DataFrame, violations_df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on establishment history."""
        
        # Calculate historical violation rates by establishment
        est_history = violations_df.groupby('inspection_id').agg({
            'is_serious': 'sum',
            'is_repeat': 'sum',
            'penalty_amount': 'sum'
        }).reset_index()
        
        est_history = est_history.merge(
            inspections_df[['inspection_id', 'establishment_id', 'open_date']], 
            on='inspection_id'
        )
        
        # Calculate rolling averages for each establishment
        est_history = est_history.sort_values(['establishment_id', 'open_date'])
        
        est_history['est_avg_serious_violations'] = est_history.groupby('establishment_id')['is_serious'].expanding().mean().reset_index(0, drop=True)
        est_history['est_avg_penalty'] = est_history.groupby('establishment_id')['penalty_amount'].expanding().mean().reset_index(0, drop=True)
        
        # Merge back with main dataframe
        history_features = est_history[['inspection_id', 'est_avg_serious_violations', 'est_avg_penalty']]
        inspections_df = inspections_df.merge(history_features, on='inspection_id', how='left')
        
        # Fill NaN values for first-time establishments
        inspections_df['est_avg_serious_violations'] = inspections_df['est_avg_serious_violations'].fillna(0)
        inspections_df['est_avg_penalty'] = inspections_df['est_avg_penalty'].fillna(0)
        
        return inspections_df
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the target variable for prediction.
        
        Args:
            df: Features dataframe
            
        Returns:
            DataFrame with target variable added
        """
        target_definition = self.config.get('models', {}).get('target_variable', {}).get('definition', 'binary_injury')
        
        if target_definition == 'binary_injury':
            # Binary classification: injury vs no injury
            df['target'] = df['has_injury'].astype(int)
        
        elif target_definition == 'severity_multiclass':
            # Multi-class: no injury, minor injury, major injury/fatality
            df['target'] = 0  # No injury
            df.loc[df['has_injury'] == True, 'target'] = 1  # Injury
            # Note: In real implementation, you'd have severity information to distinguish major injuries
        
        logger.info(f"Target variable distribution:\n{df['target'].value_counts()}")
        
        return df
    
    def prepare_ml_dataset(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare the dataset for machine learning.
        
        Args:
            features_df: Features dataframe with target variable
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Preparing ML dataset...")
        
        # Select features for ML
        feature_columns = [
            'total_violations', 'total_penalty_amount', 'serious_violation_rate',
            'repeat_violation_flag', 'willful_violation_flag', 'violation_severity_score',
            'days_since_last_inspection', 'inspections_per_year', 'industry_risk_score',
            'high_hazard_industry_flag', 'month', 'quarter', 'day_of_week',
            'is_summer', 'is_winter', 'est_avg_serious_violations', 'est_avg_penalty'
        ]
        
        # Filter to columns that exist
        available_features = [col for col in feature_columns if col in features_df.columns]
        
        # Handle categorical variables
        categorical_features = ['naics_2_digit']
        for cat_feature in categorical_features:
            if cat_feature in features_df.columns:
                if cat_feature not in self.label_encoders:
                    self.label_encoders[cat_feature] = LabelEncoder()
                    features_df[f'{cat_feature}_encoded'] = self.label_encoders[cat_feature].fit_transform(features_df[cat_feature].astype(str))
                else:
                    features_df[f'{cat_feature}_encoded'] = self.label_encoders[cat_feature].transform(features_df[cat_feature].astype(str))
                
                available_features.append(f'{cat_feature}_encoded')
        
        # Prepare X and y
        X = features_df[available_features].fillna(0).values
        y = features_df['target'].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        logger.info(f"ML dataset prepared: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Target distribution: {np.bincount(y)}")
        
        return X, y, available_features
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets.
        
        Args:
            X: Features
            y: Target variable
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        config = self.config['data_processing']['validation']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config['test_size'],
            random_state=config['random_state'],
            stratify=y if config['stratify'] else None
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train: np.ndarray, X_test: np.ndarray, 
                          y_train: np.ndarray, y_test: np.ndarray, 
                          feature_names: List[str]) -> None:
        """Save processed data to files."""
        
        # Create processed data directory
        self.processed_dir.mkdir(exist_ok=True)
        
        # Save numpy arrays
        np.save(self.processed_dir / 'X_train.npy', X_train)
        np.save(self.processed_dir / 'X_test.npy', X_test)
        np.save(self.processed_dir / 'y_train.npy', y_train)
        np.save(self.processed_dir / 'y_test.npy', y_test)
        
        # Save feature names
        with open(self.processed_dir / 'feature_names.txt', 'w') as f:
            for name in feature_names:
                f.write(f"{name}\n")
        
        # Save data summary
        summary = {
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'features': len(feature_names),
            'target_distribution_train': {
                'class_0': int(np.sum(y_train == 0)),
                'class_1': int(np.sum(y_train == 1))
            },
            'processing_date': datetime.now().isoformat()
        }
        
        import json
        with open(self.processed_dir / 'data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Processed data saved to {self.processed_dir}")

def main():
    """Main function to process OSHA data."""
    processor = OSHADataProcessor()
    
    try:
        # Load raw data
        inspections_df, violations_df = processor.load_raw_data()
        
        # Clean data
        inspections_clean, violations_clean = processor.clean_data(inspections_df, violations_df)
        
        # Engineer features
        features_df = processor.engineer_features(inspections_clean, violations_clean)
        
        # Create target variable
        features_df = processor.create_target_variable(features_df)
        
        # Prepare ML dataset
        X, y, feature_names = processor.prepare_ml_dataset(features_df)
        
        # Split data
        X_train, X_test, y_train, y_test = processor.split_data(X, y)
        
        # Save processed data
        processor.save_processed_data(X_train, X_test, y_train, y_test, feature_names)
        
        logger.info("Data processing completed successfully")
        
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 