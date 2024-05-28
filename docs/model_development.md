# Model Development Guide

## Overview

This guide details the development of the ensemble machine learning model for predicting workplace injury risk. Our approach combines Random Forest and XGBoost algorithms to achieve 84%+ accuracy in identifying high-risk establishments.

## Model Architecture

### Ensemble Approach

The platform uses a sophisticated ensemble methodology combining two powerful algorithms:

```python
from src.models.ensemble_model import SafetyEnsembleModel

# Initialize ensemble with both algorithms
ensemble = SafetyEnsembleModel(
    ensemble_method='stacking',  # or 'voting'
    base_models=['random_forest', 'xgboost']
)
```

#### 1. Random Forest Component
- **Purpose**: Captures complex feature interactions and provides robust baseline predictions
- **Hyperparameters**: 100-500 trees, max depth 10-20, min samples split 5-20
- **Strengths**: Handles missing values well, provides feature importance rankings

#### 2. XGBoost Component  
- **Purpose**: Gradient boosting for sequential error correction and fine-tuned predictions
- **Hyperparameters**: Learning rate 0.01-0.3, max depth 3-10, subsample 0.8-1.0
- **Strengths**: Superior performance on structured data, built-in regularization

#### 3. Meta-Learner (Stacking)
- **Algorithm**: Logistic Regression or Light GBM
- **Purpose**: Learns optimal combination of base model predictions
- **Cross-validation**: 5-fold stratified CV to prevent overfitting

### Target Variable Definition

```python
def create_target_variable(df):
    """
    Define high-risk establishments based on multiple criteria
    """
    df['high_risk'] = (
        (df['serious_violations'] >= 2) |
        (df['willful_violations'] > 0) | 
        (df['days_away_cases'] > 0) |
        (df['total_penalty'] > df['industry_penalty_median'] * 2)
    ).astype(int)
    
    return df
```

## Model Training Pipeline

### 1. Hyperparameter Optimization

```python
import optuna

def optimize_hyperparameters(X_train, y_train):
    def objective(trial):
        # Random Forest parameters
        rf_params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 100, 500),
            'max_depth': trial.suggest_int('rf_max_depth', 10, 30),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 5, 20)
        }
        
        # XGBoost parameters  
        xgb_params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3)
        }
        
        ensemble = SafetyEnsembleModel(rf_params=rf_params, xgb_params=xgb_params)
        cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='f1')
        return cv_scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    return study.best_params
```

### 2. Model Evaluation

```python
def comprehensive_evaluation(model, X_test, y_test):
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics
```

## Model Interpretability

### SHAP Analysis

```python
import shap

def explain_model_predictions(model, X_train, X_test, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Global feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': np.abs(shap_values).mean(0)
    }).sort_values('importance', ascending=False)
    
    return feature_importance, shap_values
```

## Performance Targets

- **Overall Accuracy**: 84%+
- **Precision (High-Risk)**: 82%+
- **Recall (High-Risk)**: 79%+
- **F1-Score**: 80%+
- **AUC-ROC**: 0.87+

## Production Deployment

### Model Monitoring

```python
def monitor_model_performance(model, new_data, reference_data):
    from scipy.stats import ks_2samp
    
    # Data drift detection
    drift_results = {}
    for feature in new_data.columns:
        statistic, p_value = ks_2samp(reference_data[feature], new_data[feature])
        drift_results[feature] = {'p_value': p_value, 'drift_detected': p_value < 0.05}
    
    return drift_results
```

### Batch Prediction Pipeline

```python
def batch_predict_risk(model, establishment_data, output_path):
    risk_scores = model.predict_proba(establishment_data)[:, 1]
    risk_predictions = model.predict(establishment_data)
    
    results = pd.DataFrame({
        'establishment_id': establishment_data.index,
        'risk_score': risk_scores,
        'risk_prediction': risk_predictions,
        'prediction_date': datetime.now()
    })
    
    results.to_csv(f"{output_path}/risk_predictions.csv", index=False)
    return results
```

## Best Practices

1. **Temporal Validation**: Use time-based splits to prevent data leakage
2. **Class Imbalance**: Apply SMOTE and cost-sensitive learning
3. **Feature Engineering**: Focus on domain-specific safety indicators
4. **Model Interpretability**: Maintain SHAP analysis for regulatory compliance
5. **Continuous Monitoring**: Track performance and data drift in production
6. **Version Control**: Maintain comprehensive model versioning and metadata 