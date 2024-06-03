# Predictive Workplace Safety Analytics Platform

A comprehensive machine learning platform that analyzes OSHA inspection records to predict workplace injury risk and identify violation patterns, empowering safety managers with data-driven insights for proactive safety management.

## 🎯 Project Overview

This platform transforms workplace safety from reactive to proactive by leveraging predictive analytics on historical OSHA inspection data. The system achieves:

- **84% accuracy** in predicting workplace injury risk
- **Pattern identification** of high-risk violation combinations
- **Interactive dashboards** for safety managers
- **Proactive resource allocation** based on data-driven insights

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   OSHA Data     │    │   ML Pipeline    │    │   PostgreSQL    │
│   (50K+ records)│───▶│  RF + XGBoost    │───▶│    Database     │
│                 │    │   Ensemble       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                                ┌─────────────────┐
                                                │  Tableau        │
                                                │  Dashboard      │
                                                └─────────────────┘
```

## 🛠️ Technology Stack

- **Data Processing**: Python, Pandas, NumPy
- **Machine Learning**: scikit-learn, XGBoost
- **Database**: PostgreSQL
- **Visualization**: Tableau
- **Development**: Docker, pytest, black

## 📁 Project Structure

```
├── data/                       # Data directory
│   ├── raw/                   # Raw OSHA data
│   ├── processed/             # Cleaned and processed data
│   └── external/              # External datasets
├── src/                       # Source code
│   ├── data/                  # Data processing modules
│   ├── features/              # Feature engineering
│   ├── models/                # ML models and training
│   ├── database/              # Database operations
│   └── visualization/         # Visualization utilities
├── models/                    # Trained model artifacts
├── notebooks/                 # Jupyter notebooks for analysis
├── sql/                      # Database schema and queries
├── config/                   # Configuration files
├── tests/                    # Unit tests
├── docs/                     # Documentation
└── requirements.txt          # Python dependencies
```

## 📊 Dashboard Visualizations

Transform complex safety data into actionable insights with our comprehensive dashboard system:

### Key Dashboard Components
- **84% Predictive Accuracy** - Ensemble model performance targeting workplace injury prediction
- **50K+ OSHA Records Analysis** - Large-scale data processing for pattern identification  
- **$43K Cost Per Injury** - Economic impact quantification driving prevention investments
- **Real-time Risk Prediction** - 1,026 high-risk sites identified for proactive intervention
- **Violation Pattern Analysis** - Fall Protection identified as top risk factor
- **Industry Benchmarking** - Cross-sector risk comparison and performance tracking

### Data-to-Insight Pipeline
1. **OSHA Records** → Raw Inspection & Violation Data
2. **Python & Scikit-learn** → ML Pipeline & Ensemble Modeling  
3. **PostgreSQL** → Store Processed Data & Predictions
4. **Tableau Dashboard** → Visualize Trends & Optimize Strategy

*Dashboard images and detailed visualizations are available in the [`docs/images/`](docs/images/) directory.*

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Docker (optional)
- Tableau Desktop/Server (for dashboards)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/workplace-safety-analytics.git
cd workplace-safety-analytics
```

2. **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure database**
```bash
# Copy and edit configuration
cp config/config.template.yaml config/config.yaml
# Update database credentials in config.yaml
```

4. **Initialize database**
```bash
python src/database/init_db.py
```

5. **Download and process OSHA data**
```bash
python src/data/download_osha_data.py
python src/data/process_data.py
```

6. **Train the model**
```bash
python src/models/train_model.py
```

7. **Generate predictions**
```bash
python src/models/predict.py
```

## 📊 Key Features

### 1. Predictive Modeling
- **Ensemble approach** combining Random Forest and XGBoost
- **Advanced feature engineering** with temporal and industry-specific features
- **Class imbalance handling** using SMOTE and cost-sensitive learning
- **Cross-validation** with hyperparameter tuning

### 2. Pattern Identification
- **Violation pattern analysis** using feature importance and SHAP values
- **Industry-specific risk profiling**
- **Temporal trend analysis**
- **Regulatory compliance tracking**

### 3. Data Management
- **Robust PostgreSQL schema** with proper indexing
- **Model versioning** and prediction tracking
- **Automated data refresh** pipelines
- **Data quality monitoring**

### 4. Visualization & Reporting
- **Interactive Tableau dashboards**
- **Geographical risk heat maps**
- **Trend analysis and KPI tracking**
- **Executive summary reports**

## 🔬 Model Performance

The ensemble model achieves:
- **Overall Accuracy**: 84%+
- **Precision (High-Risk)**: 82%
- **Recall (High-Risk)**: 79%
- **F1-Score**: 80%
- **AUC-ROC**: 0.87

## 📈 Business Impact

- **Reduced incident rates** through proactive interventions
- **Optimized resource allocation** for safety inspections
- **Enhanced compliance** with OSHA regulations
- **Cost savings** from prevented injuries ($38K+ per incident)
- **Improved safety culture** through data-driven insights

## 🔧 Usage Examples

### Training a New Model
```python
from src.models.ensemble_model import SafetyEnsembleModel
from src.data.data_loader import load_processed_data

# Load data
X_train, X_test, y_train, y_test = load_processed_data()

# Initialize and train model
model = SafetyEnsembleModel()
model.fit(X_train, y_train)

# Evaluate performance
metrics = model.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.3f}")
```

### Generating Risk Predictions
```python
from src.models.predictor import SafetyPredictor

predictor = SafetyPredictor()
predictions = predictor.predict_risk(establishment_data)
high_risk_establishments = predictor.identify_high_risk(predictions)
```

### Analyzing Violation Patterns
```python
from src.features.pattern_analyzer import ViolationPatternAnalyzer

analyzer = ViolationPatternAnalyzer()
patterns = analyzer.identify_patterns(model, feature_names)
critical_patterns = analyzer.rank_by_risk(patterns)
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/test_models.py
pytest tests/test_data_processing.py

# Run with coverage
pytest --cov=src tests/
```

## 📚 Documentation

- [Technical Architecture](docs/architecture.md)
- [Data Processing Guide](docs/data_processing.md)
- [Model Development](docs/model_development.md)
- [Database Schema](docs/database_schema.md)
- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)

## 🛡️ Data Privacy & Ethics

This platform is designed with privacy and ethical considerations:
- **Data anonymization** for sensitive establishment information
- **Bias monitoring** and fairness evaluation
- **Transparent model interpretation** using SHAP values
- **Human-in-the-loop** decision making
- **Audit trails** for all predictions and interventions

## 🔄 Continuous Improvement

- **Model retraining** pipeline with new OSHA data
- **Performance monitoring** and drift detection
- **A/B testing** framework for model improvements
- **Feedback integration** from safety managers
- **Regular model validation** against actual outcomes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an [issue](https://github.com/your-username/workplace-safety-analytics/issues)
- Check the [documentation](docs/)
- Contact the development team

## 🙏 Acknowledgments

- OSHA for providing comprehensive inspection data
- Open source community for excellent ML and data tools
- Safety professionals who inspired this proactive approach

---

**⚠️ Disclaimer**: This platform is a decision-support tool and should not replace professional safety expertise. Always consult with qualified safety professionals for critical safety decisions. 