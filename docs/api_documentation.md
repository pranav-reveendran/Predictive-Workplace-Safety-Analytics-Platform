# API Documentation

## Overview

The Predictive Workplace Safety Analytics Platform provides RESTful APIs for accessing predictions, violation patterns, and analytical insights. All APIs use JSON for data exchange and include comprehensive error handling.

## Base URL

```
https://api.workplace-safety-analytics.com/v1
```

## Authentication

All API requests require authentication using API keys in the request header:

```http
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

## Endpoints

### 1. Risk Predictions

#### Get Risk Prediction for Establishment

```http
GET /predictions/{establishment_id}
```

**Parameters:**
- `establishment_id` (required): Unique establishment identifier

**Response:**
```json
{
  "establishment_id": "EST123456",
  "prediction_date": "2024-06-02",
  "risk_score": 0.8245,
  "risk_category": "High",
  "confidence_score": 0.9156,
  "model_version": "v2.1.0",
  "feature_importance": {
    "days_since_last_inspection": 0.2341,
    "serious_violations_history": 0.1987,
    "industry_risk_score": 0.1654
  },
  "recommendations": [
    "Schedule immediate safety inspection",
    "Focus on fall protection compliance",
    "Review employee training programs"
  ]
}
```

#### Batch Risk Predictions

```http
POST /predictions/batch
```

**Request Body:**
```json
{
  "establishment_ids": ["EST123456", "EST789012", "EST345678"],
  "include_explanations": true,
  "risk_threshold": 0.7
}
```

**Response:**
```json
{
  "predictions": [
    {
      "establishment_id": "EST123456",
      "risk_score": 0.8245,
      "risk_category": "High"
    }
  ],
  "summary": {
    "total_establishments": 3,
    "high_risk_count": 1,
    "processing_time_ms": 245
  }
}
```

### 2. Establishment Data

#### Get Establishment Details

```http
GET /establishments/{establishment_id}
```

**Response:**
```json
{
  "establishment_id": "EST123456",
  "name": "ABC Manufacturing Corp",
  "naics_code": "336",
  "naics_description": "Transportation Equipment Manufacturing",
  "address": {
    "street": "123 Industrial Blvd",
    "city": "Detroit",
    "state": "MI",
    "zip_code": "48201"
  },
  "employee_count": 450,
  "establishment_type": "Manufacturing",
  "safety_metrics": {
    "last_inspection_date": "2023-11-15",
    "total_inspections": 12,
    "serious_violations_count": 5,
    "average_penalty": 8750.50
  }
}
```

#### Search Establishments

```http
GET /establishments?state=MI&naics_code=336&risk_category=High&limit=50
```

**Query Parameters:**
- `state`: State abbreviation (optional)
- `naics_code`: Industry code (optional)  
- `risk_category`: Low, Medium, High, Critical (optional)
- `limit`: Number of results (default: 20, max: 100)
- `offset`: Pagination offset (default: 0)

### 3. Violation Patterns

#### Get Violation Patterns

```http
GET /patterns
```

**Response:**
```json
{
  "patterns": [
    {
      "pattern_id": "PAT001",
      "pattern_name": "Fall Protection Deficiencies",
      "description": "Common pattern in construction and manufacturing involving inadequate fall protection systems",
      "industry_codes": ["238", "336", "331"],
      "violation_standards": ["1926.501", "1926.503", "1910.23"],
      "frequency_score": 0.7834,
      "severity_score": 0.8912,
      "prevention_recommendations": [
        "Implement comprehensive fall protection training",
        "Regular inspection of safety equipment",
        "Update safety procedures quarterly"
      ]
    }
  ]
}
```

#### Get Pattern by Industry

```http
GET /patterns/industry/{naics_code}
```

**Response:**
```json
{
  "naics_code": "336",
  "industry_name": "Transportation Equipment Manufacturing", 
  "patterns": [
    {
      "pattern_name": "Machinery Guarding Issues",
      "frequency_score": 0.6543,
      "risk_elevation": 2.34
    }
  ]
}
```

### 4. Analytics and Insights

#### Industry Benchmarks

```http
GET /analytics/benchmarks?naics_code=336
```

**Response:**
```json
{
  "naics_code": "336",
  "industry_name": "Transportation Equipment Manufacturing",
  "benchmarks": {
    "average_risk_score": 0.6432,
    "injury_rate": 0.0234,
    "average_penalty": 12543.67,
    "common_violations": [
      {
        "standard": "1910.147",
        "description": "Lockout/Tagout",
        "frequency": 156
      }
    ]
  },
  "percentiles": {
    "risk_score_p50": 0.6234,
    "risk_score_p90": 0.8567,
    "penalty_p50": 8900.00,
    "penalty_p90": 23456.78
  }
}
```

#### Temporal Trends

```http
GET /analytics/trends?start_date=2023-01-01&end_date=2024-01-01&granularity=month
```

**Response:**
```json
{
  "period": {
    "start_date": "2023-01-01",
    "end_date": "2024-01-01"
  },
  "trends": [
    {
      "month": "2023-01",
      "average_risk_score": 0.6234,
      "high_risk_count": 1026,
      "total_predictions": 4567
    }
  ],
  "summary": {
    "trend_direction": "decreasing",
    "change_rate": -0.0234,
    "statistical_significance": 0.95
  }
}
```

### 5. Model Information

#### Get Model Status

```http
GET /models/status
```

**Response:**
```json
{
  "current_model": {
    "version": "v2.1.0",
    "deployed_date": "2024-05-15T10:30:00Z",
    "accuracy": 0.8456,
    "precision": 0.8234,
    "recall": 0.7987,
    "f1_score": 0.8109
  },
  "training_data": {
    "records_count": 52340,
    "last_update": "2024-05-10T00:00:00Z",
    "coverage_period": "2020-01-01 to 2024-04-30"
  },
  "performance_monitoring": {
    "data_drift_detected": false,
    "performance_degradation": false,
    "last_validation": "2024-06-01T12:00:00Z"
  }
}
```

#### Model Metrics

```http
GET /models/metrics
```

**Response:**
```json
{
  "performance_metrics": {
    "accuracy": 0.8456,
    "precision_high_risk": 0.8234,
    "recall_high_risk": 0.7987,
    "f1_score": 0.8109,
    "auc_roc": 0.8765
  },
  "business_metrics": {
    "prevented_injuries_estimate": 234,
    "cost_savings_estimate": 10062000,
    "inspection_efficiency": 0.7234
  },
  "feature_importance": [
    {
      "feature": "days_since_last_inspection",
      "importance": 0.2341
    },
    {
      "feature": "serious_violations_history", 
      "importance": 0.1987
    }
  ]
}
```

## Webhooks

### Prediction Alerts

Register webhooks to receive real-time alerts for high-risk predictions:

```http
POST /webhooks/register
```

**Request Body:**
```json
{
  "url": "https://your-app.com/webhook/safety-alerts",
  "events": ["high_risk_prediction", "pattern_detected"],
  "risk_threshold": 0.8,
  "secret": "your_webhook_secret"
}
```

**Webhook Payload Example:**
```json
{
  "event": "high_risk_prediction",
  "timestamp": "2024-06-02T14:30:00Z",
  "data": {
    "establishment_id": "EST123456",
    "risk_score": 0.8567,
    "risk_category": "High",
    "previous_risk_score": 0.6234
  }
}
```

## Rate Limiting

- **Standard Plan**: 1,000 requests per hour
- **Premium Plan**: 10,000 requests per hour  
- **Enterprise Plan**: Unlimited

Rate limit headers included in responses:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 950
X-RateLimit-Reset: 1625097600
```

## Error Handling

### HTTP Status Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Invalid or missing API key
- `403 Forbidden`: Access denied for requested resource
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_ESTABLISHMENT_ID",
    "message": "The specified establishment ID does not exist",
    "details": {
      "establishment_id": "EST999999",
      "suggestion": "Verify the establishment ID format"
    },
    "timestamp": "2024-06-02T14:30:00Z",
    "request_id": "req_123456789"
  }
}
```

## SDKs and Libraries

### Python SDK

```bash
pip install workplace-safety-analytics
```

```python
from safety_analytics import SafetyClient

client = SafetyClient(api_key='your_api_key')

# Get risk prediction
prediction = client.get_prediction('EST123456')
print(f"Risk Score: {prediction.risk_score}")

# Batch predictions
predictions = client.batch_predict(['EST123456', 'EST789012'])
```

### JavaScript SDK

```bash
npm install @workplace-safety/analytics-js
```

```javascript
const SafetyAnalytics = require('@workplace-safety/analytics-js');

const client = new SafetyAnalytics('your_api_key');

// Get establishment data
client.getEstablishment('EST123456')
  .then(establishment => {
    console.log('Company:', establishment.name);
  })
  .catch(error => {
    console.error('Error:', error.message);
  });
```

## Examples

### Get High-Risk Establishments

```bash
curl -X GET \
  "https://api.workplace-safety-analytics.com/v1/establishments?risk_category=High&state=CA&limit=10" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json"
```

### Monitor Industry Trends

```bash
curl -X GET \
  "https://api.workplace-safety-analytics.com/v1/analytics/trends?naics_code=336&granularity=month" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Batch Risk Assessment

```bash
curl -X POST \
  "https://api.workplace-safety-analytics.com/v1/predictions/batch" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "establishment_ids": ["EST123456", "EST789012"],
    "include_explanations": true
  }'
```

## Changelog

### v1.2.0 (2024-06-01)
- Added violation pattern endpoints
- Enhanced prediction explanations with SHAP values
- Improved error handling and validation

### v1.1.0 (2024-05-15)  
- Added webhook support for real-time alerts
- New batch prediction endpoint
- Performance optimizations

### v1.0.0 (2024-05-01)
- Initial API release
- Core prediction and analytics endpoints
- Authentication and rate limiting 