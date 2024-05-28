# Data Processing Guide

## Overview

This guide covers the comprehensive data processing pipeline for the Predictive Workplace Safety Analytics Platform. Our system processes 50K+ OSHA inspection records to create high-quality features for machine learning models.

## Data Sources

### OSHA Inspection Data
- **Source**: OSHA Enforcement Database
- **Format**: JSON/CSV via OSHA API
- **Volume**: 50,000+ inspection records
- **Update Frequency**: Weekly
- **Coverage**: All U.S. establishments with OSHA inspections

### Key Data Fields
- **Establishment Information**: Name, address, industry code (NAICS), employee count
- **Inspection Details**: Date, type, scope, duration, inspector ID
- **Violations**: Citation type, standard violated, severity, penalty amount
- **Outcomes**: Injuries, illnesses, fatalities, days away from work

## Data Processing Pipeline

### 1. Data Acquisition (`src/data/download_osha_data.py`)

```python
# Download OSHA data with date range
downloader = OSHADataDownloader()
data = downloader.download_inspection_data(
    start_date="2020-01-01",
    end_date="2024-01-01"
)
```

**Features:**
- Incremental data loading
- API rate limiting and retry logic
- Data validation and quality checks
- Progress tracking for large datasets

### 2. Data Cleaning

**Missing Value Handling:**
- Employee count: Impute using industry medians
- Penalty amounts: Use regulatory minimums for missing values
- Geographic data: Geocode addresses using external APIs

**Outlier Detection:**
- Statistical methods (IQR, Z-score) for numerical features
- Domain knowledge filters (e.g., penalty amounts > $1M flagged)
- Temporal anomalies (inspections on holidays/weekends)

**Data Standardization:**
- NAICS code normalization to 3-digit industry groups
- Address standardization and geocoding
- Date/time formatting and timezone handling

### 3. Feature Engineering

#### Temporal Features
```python
def create_temporal_features(df):
    df['days_since_last_inspection'] = calculate_days_since_last(df)
    df['inspection_frequency'] = calculate_inspection_frequency(df)
    df['seasonal_indicator'] = extract_season(df['inspection_date'])
    df['day_of_week'] = df['inspection_date'].dt.dayofweek
    return df
```

**Generated Features:**
- Days since last inspection (establishment level)
- Inspection frequency (annual rate)
- Seasonal patterns (Q1-Q4 indicators)
- Day of week effects
- Time since establishment opening

#### Industry-Specific Features
```python
def create_industry_features(df):
    df['industry_risk_score'] = calculate_industry_risk(df['naics_code'])
    df['industry_avg_penalty'] = calculate_industry_penalty_avg(df)
    df['relative_employee_count'] = normalize_by_industry(df)
    return df
```

**Generated Features:**
- Industry risk benchmarks
- Relative establishment size within industry
- Historical industry violation patterns
- Regulatory focus indicators by industry

#### Violation-Based Features
```python
def create_violation_features(df):
    df['violation_severity_score'] = calculate_severity_score(df)
    df['repeat_violation_indicator'] = identify_repeat_violations(df)
    df['violation_type_diversity'] = count_violation_types(df)
    return df
```

**Generated Features:**
- Violation severity scoring (weighted by type and penalty)
- Repeat violation indicators
- Violation type diversity index
- Critical violation flags (willful, repeat, serious)

#### Geographic Features
```python
def create_geographic_features(df):
    df['regional_risk_score'] = calculate_regional_risk(df)
    df['urban_rural_indicator'] = classify_location(df)
    df['state_regulatory_strength'] = map_state_regulations(df)
    return df
```

**Generated Features:**
- Regional risk scores by state/county
- Urban vs. rural classification
- State regulatory environment strength
- Local economic indicators

### 4. Target Variable Engineering

#### Binary Classification Target
```python
def create_injury_risk_target(df):
    # High risk defined as serious injury within 12 months
    df['high_injury_risk'] = (
        (df['days_away_cases'] > 0) |
        (df['serious_violations'] > 2) |
        (df['willful_violations'] > 0)
    )
    return df
```

#### Multi-class Target (Optional)
- **Low Risk**: No violations or minor citations only
- **Medium Risk**: Serious violations but no injuries
- **High Risk**: Serious violations with injury history
- **Critical Risk**: Willful violations or fatalities

### 5. Data Splitting and Validation

#### Temporal Splitting Strategy
```python
def temporal_split(df, train_end_date, val_end_date):
    train_data = df[df['inspection_date'] <= train_end_date]
    val_data = df[
        (df['inspection_date'] > train_end_date) & 
        (df['inspection_date'] <= val_end_date)
    ]
    test_data = df[df['inspection_date'] > val_end_date]
    return train_data, val_data, test_data
```

**Rationale:**
- Prevents data leakage by maintaining temporal order
- Simulates real-world prediction scenarios
- Enables model performance validation over time

#### Cross-Validation Strategy
- **Time Series Split**: 5-fold temporal cross-validation
- **Stratified Sampling**: Maintain class balance across folds
- **Group-based Splitting**: Prevent establishment data leakage

### 6. Class Imbalance Handling

#### SMOTE (Synthetic Minority Oversampling)
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=5)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

#### Cost-Sensitive Learning
```python
# Calculate class weights for imbalanced data
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
```

### 7. Feature Selection

#### Statistical Methods
- **Chi-square test** for categorical features
- **ANOVA F-test** for numerical features
- **Mutual information** for feature relevance

#### Model-Based Selection
```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# Feature importance-based selection
rf_selector = SelectFromModel(
    RandomForestClassifier(n_estimators=100),
    threshold='median'
)
X_selected = rf_selector.fit_transform(X_train, y_train)
```

#### Correlation Analysis
- Remove highly correlated features (>0.95)
- Variance inflation factor (VIF) analysis
- Principal component analysis for dimensionality reduction

### 8. Data Quality Monitoring

#### Automated Checks
```python
def validate_data_quality(df):
    checks = {
        'missing_values': check_missing_values(df),
        'outliers': detect_outliers(df),
        'duplicates': find_duplicates(df),
        'schema_compliance': validate_schema(df),
        'referential_integrity': check_foreign_keys(df)
    }
    return checks
```

#### Data Drift Detection
- Statistical tests for feature distribution changes
- KL divergence monitoring
- Population stability index (PSI)

### 9. Performance Optimization

#### Memory Management
```python
# Optimize data types for memory efficiency
def optimize_dtypes(df):
    for col in df.select_dtypes(include=['int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df
```

#### Parallel Processing
```python
from multiprocessing import Pool
import numpy as np

def process_chunk(chunk):
    return apply_feature_engineering(chunk)

# Process data in parallel chunks
with Pool(processes=4) as pool:
    results = pool.map(process_chunk, data_chunks)
```

## Configuration

### Processing Parameters
```yaml
data_processing:
  chunk_size: 10000
  missing_value_threshold: 0.3
  outlier_method: "iqr"
  outlier_threshold: 3.0
  correlation_threshold: 0.95
  
feature_engineering:
  temporal_window_days: 365
  industry_grouping_level: 3  # NAICS digits
  geographic_level: "county"
  
validation:
  test_size: 0.2
  validation_size: 0.1
  cv_folds: 5
  stratify: true
```

## Usage Examples

### Complete Processing Pipeline
```python
from src.data.process_data import DataProcessor

# Initialize processor
processor = DataProcessor(config_path='config/config.yaml')

# Load raw data
raw_data = processor.load_raw_data('data/raw/osha_inspections.csv')

# Apply full processing pipeline
processed_data = processor.process_data(raw_data)

# Save processed data
processor.save_processed_data(processed_data, 'data/processed/')
```

### Custom Feature Engineering
```python
# Add custom features
def custom_safety_score(df):
    df['safety_culture_score'] = (
        df['voluntary_compliance'] * 0.3 +
        df['training_programs'] * 0.2 +
        df['safety_investment'] * 0.5
    )
    return df

processor.add_feature_transformer(custom_safety_score)
```

## Best Practices

### 1. Data Lineage
- Track data transformations and their impact
- Maintain audit logs for compliance
- Version control for processing scripts

### 2. Reproducibility
- Set random seeds for all stochastic operations
- Use deterministic algorithms where possible
- Document all processing decisions

### 3. Scalability
- Design for incremental processing
- Use efficient data formats (Parquet, HDF5)
- Implement streaming processing for real-time updates

### 4. Validation
- Implement comprehensive unit tests
- Use data validation frameworks (Great Expectations)
- Monitor data quality metrics over time

## Troubleshooting

### Common Issues

**Memory Errors:**
- Reduce chunk size in processing configuration
- Use data type optimization
- Implement out-of-core processing with Dask

**Performance Issues:**
- Profile code to identify bottlenecks
- Use vectorized operations instead of loops
- Consider GPU acceleration for large datasets

**Data Quality Issues:**
- Implement robust error handling
- Add comprehensive logging
- Use data validation checkpoints

## Future Enhancements

1. **Real-time Processing**: Stream processing with Apache Kafka
2. **Advanced Feature Engineering**: Deep learning-based feature extraction
3. **Automated Feature Selection**: AutoML feature engineering
4. **Data Lake Integration**: Connect to enterprise data lakes
5. **Edge Computing**: Distributed processing for large-scale deployments 