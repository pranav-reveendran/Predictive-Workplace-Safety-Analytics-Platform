"""
Model Training Script

This script trains the ensemble model for workplace safety prediction using processed OSHA data.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
import json
import yaml
from typing import List, Dict

# Local imports
from ensemble_model import SafetyEnsembleModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Handles the training of the workplace safety prediction model."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the trainer with configuration."""
        self.config_path = config_path
        self.data_dir = Path("data/processed")
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
    def load_processed_data(self) -> tuple:
        """
        Load processed training and test data.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        logger.info("Loading processed data...")
        
        try:
            # Load numpy arrays
            X_train = np.load(self.data_dir / 'X_train.npy')
            X_test = np.load(self.data_dir / 'X_test.npy')
            y_train = np.load(self.data_dir / 'y_train.npy')
            y_test = np.load(self.data_dir / 'y_test.npy')
            
            # Load feature names
            with open(self.data_dir / 'feature_names.txt', 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
            
            logger.info(f"Data loaded successfully:")
            logger.info(f"  Training set: {X_train.shape}")
            logger.info(f"  Test set: {X_test.shape}")
            logger.info(f"  Features: {len(feature_names)}")
            
            return X_train, X_test, y_train, y_test, feature_names
            
        except FileNotFoundError as e:
            logger.error(f"Processed data not found: {str(e)}")
            logger.error("Please run data processing first: python src/data/process_data.py")
            raise
    
    def train_model(self) -> SafetyEnsembleModel:
        """
        Train the ensemble model.
        
        Returns:
            Trained SafetyEnsembleModel
        """
        logger.info("Starting model training process...")
        
        # Load data
        X_train, X_test, y_train, y_test, feature_names = self.load_processed_data()
        
        # Initialize model
        model = SafetyEnsembleModel(config_path=self.config_path)
        
        # Perform cross-validation before final training
        logger.info("Performing cross-validation...")
        cv_scores = model.cross_validate(X_train, y_train)
        
        # Train the model
        logger.info("Training ensemble model...")
        model.fit(X_train, y_train, feature_names=feature_names)
        
        # Evaluate on test set
        logger.info("Evaluating model performance...")
        test_metrics = model.evaluate(X_test, y_test)
        
        # Identify violation patterns
        logger.info("Identifying violation patterns...")
        violation_patterns = model.identify_violation_patterns(feature_names)
        
        # Save model and results
        self.save_model_and_results(model, test_metrics, cv_scores, violation_patterns, feature_names)
        
        return model
    
    def save_model_and_results(self, model: SafetyEnsembleModel, 
                              test_metrics: Dict, cv_scores: Dict,
                              violation_patterns: List[Dict], 
                              feature_names: List[str]):
        """Save the trained model and results."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.models_dir / f"safety_ensemble_model_{timestamp}.pkl"
        
        metadata = {
            'training_date': datetime.now().isoformat(),
            'test_metrics': test_metrics,
            'cv_scores': cv_scores,
            'feature_count': len(feature_names),
            'violation_patterns_count': len(violation_patterns)
        }
        
        model.save_model(str(model_path), metadata=metadata)
        
        # Save detailed results
        results = {
            'model_info': {
                'model_type': 'SafetyEnsembleModel',
                'training_date': datetime.now().isoformat(),
                'model_path': str(model_path),
                'config_path': self.config_path
            },
            'performance_metrics': {
                'test_metrics': test_metrics,
                'cross_validation': cv_scores
            },
            'feature_analysis': {
                'feature_names': feature_names,
                'feature_importance': model.feature_importance
            },
            'violation_patterns': violation_patterns,
            'model_targets_achieved': self._check_targets_achieved(test_metrics)
        }
        
        results_path = self.models_dir / f"training_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Results saved to: {results_path}")
        
        # Create latest model symlink
        latest_model_path = self.models_dir / "latest_model.pkl"
        if latest_model_path.exists():
            latest_model_path.unlink()
        
        # Create a relative symlink
        latest_model_path.symlink_to(model_path.name)
        
        # Save feature importance summary
        self._save_feature_importance_summary(model.feature_importance, feature_names, timestamp)
        
        # Save violation patterns summary
        self._save_violation_patterns_summary(violation_patterns, timestamp)
    
    def _check_targets_achieved(self, metrics: Dict) -> Dict[str, bool]:
        """Check which performance targets were achieved."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except:
            return {}
        
        targets = config.get('models', {}).get('performance_targets', {})
        achieved = {}
        
        for target_name, target_value in targets.items():
            actual_value = metrics.get(target_name, 0)
            achieved[target_name] = actual_value >= target_value
        
        return achieved
    
    def _save_feature_importance_summary(self, feature_importance: Dict, 
                                       feature_names: List[str], timestamp: str):
        """Save a summary of feature importance."""
        if not feature_importance.get('ensemble_average'):
            return
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature_name': feature_names,
            'random_forest_importance': feature_importance['random_forest'],
            'xgboost_importance': feature_importance['xgboost'],
            'ensemble_average_importance': feature_importance['ensemble_average']
        })
        
        # Sort by ensemble average importance
        importance_df = importance_df.sort_values('ensemble_average_importance', ascending=False)
        
        # Save to CSV
        importance_path = self.models_dir / f"feature_importance_{timestamp}.csv"
        importance_df.to_csv(importance_path, index=False)
        
        logger.info(f"Feature importance saved to: {importance_path}")
    
    def _save_violation_patterns_summary(self, patterns: List[Dict], timestamp: str):
        """Save violation patterns to CSV for easy analysis."""
        if not patterns:
            return
        
        patterns_df = pd.DataFrame(patterns)
        patterns_path = self.models_dir / f"violation_patterns_{timestamp}.csv"
        patterns_df.to_csv(patterns_path, index=False)
        
        logger.info(f"Violation patterns saved to: {patterns_path}")
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Perform hyperparameter tuning for the ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Best hyperparameters found
        """
        logger.info("Starting hyperparameter tuning...")
        
        # This is a simplified version - in practice, you might want to use more sophisticated methods
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb
        
        # Define parameter grids
        rf_param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        xgb_param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 6, 9],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        best_params = {}
        
        # Tune Random Forest
        logger.info("Tuning Random Forest parameters...")
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
        rf_grid.fit(X_train, y_train)
        best_params['random_forest'] = rf_grid.best_params_
        
        # Tune XGBoost
        logger.info("Tuning XGBoost parameters...")
        xgb_model = xgb.XGBClassifier(random_state=42)
        xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
        xgb_grid.fit(X_train, y_train)
        best_params['xgboost'] = xgb_grid.best_params_
        
        logger.info("Hyperparameter tuning completed")
        logger.info(f"Best RF params: {best_params['random_forest']}")
        logger.info(f"Best XGB params: {best_params['xgboost']}")
        
        return best_params

def main():
    """Main training function."""
    logger.info("=== Workplace Safety Analytics Model Training ===")
    
    trainer = ModelTrainer()
    
    try:
        # Train the model
        model = trainer.train_model()
        
        logger.info("Training completed successfully!")
        logger.info("Model is ready for predictions and deployment.")
        
        # Optional: Perform additional analysis
        logger.info("Training process completed. Check the models/ directory for saved artifacts.")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 