# Database Schema Documentation

## Overview

The Predictive Workplace Safety Analytics Platform uses a PostgreSQL database to store OSHA inspection data, model predictions, and violation patterns. The schema is optimized for analytical queries and supports the ML pipeline requirements.

## Schema Design Principles

- **Normalized Structure**: Reduces data redundancy while maintaining query performance
- **Indexed Columns**: Strategic indexing for fast analytical queries  
- **Temporal Partitioning**: Efficient storage and querying of time-series data
- **Audit Trail**: Comprehensive logging for compliance and debugging
- **Scalability**: Designed to handle 50K+ inspection records efficiently

## Core Tables

### 1. establishments

Stores information about workplaces subject to OSHA inspections.

```sql
CREATE TABLE establishments (
    establishment_id SERIAL PRIMARY KEY,
    establishment_name VARCHAR(255) NOT NULL,
    naics_code VARCHAR(10),
    naics_description TEXT,
    street_address VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(2),
    zip_code VARCHAR(10),
    employee_count INTEGER,
    establishment_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Key Fields:**
- `naics_code`: Industry classification for risk benchmarking
- `employee_count`: Used for size-based risk adjustment
- `establishment_type`: Manufacturing, construction, etc.

**Indexes:**
```sql
CREATE INDEX idx_establishments_naics ON establishments(naics_code);
CREATE INDEX idx_establishments_state ON establishments(state);
CREATE INDEX idx_establishments_size ON establishments(employee_count);
```

### 2. inspections

Central table storing OSHA inspection records.

```sql
CREATE TABLE inspections (
    inspection_id SERIAL PRIMARY KEY,
    establishment_id INTEGER REFERENCES establishments(establishment_id),
    osha_activity_number VARCHAR(50) UNIQUE,
    inspection_date DATE NOT NULL,
    inspection_type VARCHAR(50),
    inspection_scope VARCHAR(50),
    inspector_id VARCHAR(20),
    case_closed_date DATE,
    total_penalties DECIMAL(12,2),
    serious_violations INTEGER DEFAULT 0,
    willful_violations INTEGER DEFAULT 0,
    repeat_violations INTEGER DEFAULT 0,
    other_violations INTEGER DEFAULT 0,
    days_away_cases INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Key Fields:**
- `inspection_type`: Programmed, complaint, accident, referral
- `inspection_scope`: Complete, partial, records only
- `total_penalties`: Sum of all violation penalties
- `days_away_cases`: Injury severity indicator

**Indexes:**
```sql
CREATE INDEX idx_inspections_date ON inspections(inspection_date);
CREATE INDEX idx_inspections_establishment ON inspections(establishment_id);
CREATE INDEX idx_inspections_type ON inspections(inspection_type);
CREATE INDEX idx_inspections_penalties ON inspections(total_penalties);
```

### 3. violations

Detailed violation records linked to inspections.

```sql
CREATE TABLE violations (
    violation_id SERIAL PRIMARY KEY,
    inspection_id INTEGER REFERENCES inspections(inspection_id),
    citation_id VARCHAR(50),
    violation_type VARCHAR(20),
    standard_violated VARCHAR(100),
    standard_description TEXT,
    severity VARCHAR(20),
    proposed_penalty DECIMAL(10,2),
    final_penalty DECIMAL(10,2),
    abatement_date DATE,
    violation_description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Key Fields:**
- `violation_type`: Serious, willful, repeat, other
- `standard_violated`: OSHA standard code (e.g., 1926.501)
- `severity`: Categorical severity assessment
- `proposed_penalty` vs `final_penalty`: Track penalty reductions

**Indexes:**
```sql
CREATE INDEX idx_violations_inspection ON violations(inspection_id);
CREATE INDEX idx_violations_type ON violations(violation_type);
CREATE INDEX idx_violations_standard ON violations(standard_violated);
CREATE INDEX idx_violations_severity ON violations(severity);
```

### 4. model_predictions

Stores ML model outputs for risk assessment.

```sql
CREATE TABLE model_predictions (
    prediction_id SERIAL PRIMARY KEY,
    establishment_id INTEGER REFERENCES establishments(establishment_id),
    model_version VARCHAR(20),
    prediction_date DATE NOT NULL,
    risk_score DECIMAL(5,4),
    risk_category VARCHAR(20),
    confidence_score DECIMAL(5,4),
    feature_importance JSONB,
    shap_values JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Key Fields:**
- `risk_score`: Probability of high-risk classification (0-1)
- `risk_category`: Low, Medium, High, Critical
- `confidence_score`: Model confidence in prediction
- `feature_importance`: JSON storing top contributing features
- `shap_values`: JSON storing SHAP explanation values

**Indexes:**
```sql
CREATE INDEX idx_predictions_establishment ON model_predictions(establishment_id);
CREATE INDEX idx_predictions_date ON model_predictions(prediction_date);
CREATE INDEX idx_predictions_risk_score ON model_predictions(risk_score DESC);
CREATE INDEX idx_predictions_model_version ON model_predictions(model_version);
```

### 5. violation_patterns

Stores identified patterns for proactive safety management.

```sql
CREATE TABLE violation_patterns (
    pattern_id SERIAL PRIMARY KEY,
    pattern_name VARCHAR(100) NOT NULL,
    pattern_description TEXT,
    industry_codes TEXT[], -- Array of NAICS codes
    violation_standards TEXT[], -- Array of common standards
    risk_indicators JSONB,
    frequency_score DECIMAL(5,4),
    severity_score DECIMAL(5,4),
    prevention_recommendations TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Key Fields:**
- `industry_codes`: Industries where pattern is prevalent
- `violation_standards`: Common OSHA standards involved
- `risk_indicators`: Statistical measures of risk elevation
- `frequency_score`: How often pattern leads to violations
- `severity_score`: Typical severity when pattern occurs

## Views and Analytical Queries

### 1. Establishment Risk Summary View

```sql
CREATE VIEW establishment_risk_summary AS
SELECT 
    e.establishment_id,
    e.establishment_name,
    e.naics_code,
    e.state,
    e.employee_count,
    COUNT(i.inspection_id) as total_inspections,
    AVG(i.total_penalties) as avg_penalty,
    SUM(i.serious_violations) as total_serious_violations,
    SUM(i.willful_violations) as total_willful_violations,
    MAX(i.inspection_date) as last_inspection_date,
    mp.risk_score as latest_risk_score,
    mp.risk_category as latest_risk_category
FROM establishments e
LEFT JOIN inspections i ON e.establishment_id = i.establishment_id
LEFT JOIN (
    SELECT DISTINCT ON (establishment_id) 
           establishment_id, risk_score, risk_category, prediction_date
    FROM model_predictions 
    ORDER BY establishment_id, prediction_date DESC
) mp ON e.establishment_id = mp.establishment_id
GROUP BY e.establishment_id, e.establishment_name, e.naics_code, 
         e.state, e.employee_count, mp.risk_score, mp.risk_category;
```

### 2. Industry Benchmark View

```sql
CREATE VIEW industry_benchmarks AS
SELECT 
    e.naics_code,
    e.naics_description,
    COUNT(DISTINCT e.establishment_id) as establishment_count,
    AVG(i.total_penalties) as avg_penalty_per_inspection,
    AVG(i.serious_violations) as avg_serious_violations,
    AVG(CASE WHEN i.days_away_cases > 0 THEN 1.0 ELSE 0.0 END) as injury_rate,
    AVG(mp.risk_score) as avg_predicted_risk
FROM establishments e
LEFT JOIN inspections i ON e.establishment_id = i.establishment_id
LEFT JOIN model_predictions mp ON e.establishment_id = mp.establishment_id
WHERE i.inspection_date >= CURRENT_DATE - INTERVAL '2 years'
GROUP BY e.naics_code, e.naics_description
HAVING COUNT(DISTINCT e.establishment_id) >= 10;
```

### 3. Temporal Risk Trends View

```sql
CREATE VIEW monthly_risk_trends AS
SELECT 
    DATE_TRUNC('month', mp.prediction_date) as month,
    COUNT(*) as total_predictions,
    AVG(mp.risk_score) as avg_risk_score,
    COUNT(CASE WHEN mp.risk_category = 'High' THEN 1 END) as high_risk_count,
    COUNT(CASE WHEN mp.risk_category = 'Critical' THEN 1 END) as critical_risk_count
FROM model_predictions mp
WHERE mp.prediction_date >= CURRENT_DATE - INTERVAL '2 years'
GROUP BY DATE_TRUNC('month', mp.prediction_date)
ORDER BY month;
```

## Performance Optimization

### Partitioning Strategy

Large tables are partitioned by date for improved query performance:

```sql
-- Partition inspections table by year
CREATE TABLE inspections_2023 PARTITION OF inspections
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE inspections_2024 PARTITION OF inspections
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
```

### Query Optimization Examples

**High-Risk Establishments Query:**
```sql
-- Optimized query for dashboard
SELECT 
    e.establishment_name,
    e.city,
    e.state,
    mp.risk_score,
    mp.prediction_date
FROM establishments e
INNER JOIN model_predictions mp ON e.establishment_id = mp.establishment_id
WHERE mp.risk_score > 0.7 
    AND mp.prediction_date = (
        SELECT MAX(prediction_date) 
        FROM model_predictions mp2 
        WHERE mp2.establishment_id = mp.establishment_id
    )
ORDER BY mp.risk_score DESC
LIMIT 1000;
```

**Industry Risk Analysis:**
```sql
-- Violation patterns by industry
SELECT 
    e.naics_code,
    v.standard_violated,
    COUNT(*) as violation_count,
    AVG(v.final_penalty) as avg_penalty,
    COUNT(DISTINCT e.establishment_id) as affected_establishments
FROM violations v
INNER JOIN inspections i ON v.inspection_id = i.inspection_id
INNER JOIN establishments e ON i.establishment_id = e.establishment_id
WHERE i.inspection_date >= CURRENT_DATE - INTERVAL '1 year'
GROUP BY e.naics_code, v.standard_violated
HAVING COUNT(*) >= 5
ORDER BY violation_count DESC;
```

## Data Quality and Constraints

### Referential Integrity

```sql
-- Foreign key constraints ensure data consistency
ALTER TABLE inspections 
    ADD CONSTRAINT fk_inspections_establishment 
    FOREIGN KEY (establishment_id) REFERENCES establishments(establishment_id);

ALTER TABLE violations 
    ADD CONSTRAINT fk_violations_inspection 
    FOREIGN KEY (inspection_id) REFERENCES inspections(inspection_id);

ALTER TABLE model_predictions 
    ADD CONSTRAINT fk_predictions_establishment 
    FOREIGN KEY (establishment_id) REFERENCES establishments(establishment_id);
```

### Data Validation

```sql
-- Check constraints for data quality
ALTER TABLE inspections 
    ADD CONSTRAINT chk_inspection_date_valid 
    CHECK (inspection_date <= CURRENT_DATE);

ALTER TABLE violations 
    ADD CONSTRAINT chk_penalty_non_negative 
    CHECK (proposed_penalty >= 0 AND final_penalty >= 0);

ALTER TABLE model_predictions 
    ADD CONSTRAINT chk_risk_score_valid 
    CHECK (risk_score >= 0.0 AND risk_score <= 1.0);
```

## Backup and Recovery

### Backup Strategy

```sql
-- Daily backup of critical tables
pg_dump --host=localhost --port=5432 --username=postgres \
        --table=establishments --table=inspections \
        --table=violations --table=model_predictions \
        safety_analytics > backup_$(date +%Y%m%d).sql
```

### Point-in-Time Recovery

```sql
-- Enable WAL archiving for point-in-time recovery
archive_mode = on
archive_command = 'cp %p /backup/archive/%f'
```

## Security and Access Control

### User Roles

```sql
-- Read-only role for analysts
CREATE ROLE safety_analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO safety_analyst;

-- Read-write role for ML pipeline
CREATE ROLE ml_pipeline;
GRANT SELECT, INSERT, UPDATE ON model_predictions TO ml_pipeline;
GRANT SELECT ON establishments, inspections, violations TO ml_pipeline;

-- Admin role for database maintenance
CREATE ROLE safety_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO safety_admin;
```

### Row Level Security

```sql
-- Restrict access based on establishment geography
ALTER TABLE establishments ENABLE ROW LEVEL SECURITY;

CREATE POLICY state_access_policy ON establishments
    FOR ALL TO safety_analyst
    USING (state = current_setting('app.user_state'));
```

## Monitoring and Maintenance

### Performance Monitoring

```sql
-- Monitor slow queries
SELECT 
    query,
    mean_time,
    calls,
    total_time
FROM pg_stat_statements 
WHERE mean_time > 1000  -- Queries slower than 1 second
ORDER BY mean_time DESC;
```

### Index Usage Analysis

```sql
-- Identify unused indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_scan = 0;
```

## Migration Scripts

### Schema Updates

```sql
-- Example migration for adding new risk factors
ALTER TABLE establishments 
    ADD COLUMN safety_program_score DECIMAL(3,2),
    ADD COLUMN last_safety_audit_date DATE,
    ADD COLUMN voluntary_compliance_flag BOOLEAN DEFAULT FALSE;

-- Update existing records
UPDATE establishments 
SET safety_program_score = 0.5, 
    voluntary_compliance_flag = FALSE 
WHERE safety_program_score IS NULL;
```

This schema design supports the analytical requirements of the platform while maintaining data integrity, performance, and scalability for production deployment. 