"""
Tests for the ensemble model and related components.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.ensemble_model import SafetyEnsembleModel

class TestSafetyEnsembleModel:
    """Test cases for the SafetyEnsembleModel."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data
        self.n_samples = 1000
        self.n_features = 10
        
        np.random.seed(42)
        self.X = np.random.randn(self.n_samples, self.n_features)
        self.y = np.random.choice([0, 1], self.n_samples, p=[0.7, 0.3])  # Imbalanced classes
        
        self.feature_names = [f"feature_{i}" for i in range(self.n_features)]
        
        # Split data
        self.split_idx = int(0.8 * self.n_samples)
        self.X_train = self.X[:self.split_idx]
        self.X_test = self.X[self.split_idx:]
        self.y_train = self.y[:self.split_idx]
        self.y_test = self.y[self.split_idx:]
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = SafetyEnsembleModel()
        
        assert model.random_forest is not None
        assert model.xgboost is not None
        assert hasattr(model, 'performance_metrics')
        assert hasattr(model, 'feature_importance')
    
    def test_model_fitting(self):
        """Test model fitting."""
        model = SafetyEnsembleModel()
        
        # Fit the model
        model.fit(self.X_train, self.y_train, feature_names=self.feature_names)
        
        # Check that models are trained
        assert model.random_forest is not None
        assert model.xgboost is not None
        
        # Check feature importance is calculated
        assert len(model.feature_importance) > 0
        assert 'ensemble_average' in model.feature_importance
    
    def test_predictions(self):
        """Test model predictions."""
        model = SafetyEnsembleModel()
        model.fit(self.X_train, self.y_train, feature_names=self.feature_names)
        
        # Test predictions
        predictions = model.predict(self.X_test)
        probabilities = model.predict_proba(self.X_test)
        risk_categories = model.predict_risk_category(self.X_test)
        
        # Check output shapes and types
        assert len(predictions) == len(self.X_test)
        assert len(probabilities) == len(self.X_test)
        assert probabilities.shape[1] == 2  # Binary classification
        assert len(risk_categories) == len(self.X_test)
        
        # Check value ranges
        assert all(p in [0, 1] for p in predictions)
        assert all(0 <= prob <= 1 for prob_pair in probabilities for prob in prob_pair)
        assert all(cat in ['Low', 'Medium', 'High', 'Critical'] for cat in risk_categories)
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        model = SafetyEnsembleModel()
        model.fit(self.X_train, self.y_train, feature_names=self.feature_names)
        
        # Evaluate model
        metrics = model.evaluate(self.X_test, self.y_test)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc',
            'precision_high_risk', 'recall_high_risk', 'f1_score_high_risk'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1  # Metrics should be between 0 and 1
    
    def test_violation_pattern_identification(self):
        """Test violation pattern identification."""
        model = SafetyEnsembleModel()
        model.fit(self.X_train, self.y_train, feature_names=self.feature_names)
        
        patterns = model.identify_violation_patterns(self.feature_names)
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        
        # Check pattern structure
        for pattern in patterns:
            assert 'pattern_id' in pattern
            assert 'pattern_name' in pattern
            assert 'primary_feature' in pattern
            assert 'importance_score' in pattern
            assert 'pattern_type' in pattern
            assert 'risk_level' in pattern
    
    def test_cross_validation(self):
        """Test cross-validation functionality."""
        model = SafetyEnsembleModel()
        
        # This might take a while, so use smaller dataset
        X_small = self.X_train[:200]
        y_small = self.y_train[:200]
        
        cv_scores = model.cross_validate(X_small, y_small, cv_folds=3)
        
        assert isinstance(cv_scores, dict)
        assert 'accuracy_mean' in cv_scores
        assert 'accuracy_std' in cv_scores
        
        # Check that scores are reasonable
        assert 0 <= cv_scores['accuracy_mean'] <= 1
        assert cv_scores['accuracy_std'] >= 0

@pytest.fixture
def sample_data():
    """Fixture providing sample data for tests."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.choice([0, 1], 100, p=[0.7, 0.3])
    feature_names = [f"feature_{i}" for i in range(5)]
    
    return X, y, feature_names

def test_model_with_sample_data(sample_data):
    """Test model with sample data fixture."""
    X, y, feature_names = sample_data
    
    model = SafetyEnsembleModel()
    model.fit(X, y, feature_names=feature_names)
    
    predictions = model.predict(X)
    assert len(predictions) == len(X)

if __name__ == "__main__":
    pytest.main([__file__]) 