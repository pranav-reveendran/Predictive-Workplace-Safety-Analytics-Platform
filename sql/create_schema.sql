-- Predictive Workplace Safety Analytics Platform
-- PostgreSQL Database Schema

-- Drop existing tables if they exist (for development)
DROP TABLE IF EXISTS model_predictions CASCADE;
DROP TABLE IF EXISTS identified_violation_patterns CASCADE;
DROP TABLE IF EXISTS model_versions CASCADE;
DROP TABLE IF EXISTS accidents CASCADE;
DROP TABLE IF EXISTS violations CASCADE;
DROP TABLE IF EXISTS inspections CASCADE;
DROP TABLE IF EXISTS establishments CASCADE;
DROP TABLE IF EXISTS violation_types CASCADE;
DROP TABLE IF EXISTS inspection_types CASCADE;
DROP TABLE IF EXISTS inspection_scopes CASCADE;
DROP TABLE IF EXISTS naics_codes CASCADE;
DROP TABLE IF EXISTS injury_degrees CASCADE;

-- Create lookup tables first

-- NAICS Codes lookup table
CREATE TABLE naics_codes (
    naics_code VARCHAR(10) PRIMARY KEY,
    naics_title VARCHAR(255) NOT NULL,
    naics_2_digit VARCHAR(2),
    naics_4_digit VARCHAR(4),
    sector_name VARCHAR(100),
    high_hazard_flag BOOLEAN DEFAULT FALSE,
    industry_risk_score DECIMAL(3,2) DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Inspection Types lookup table
CREATE TABLE inspection_types (
    inspection_type_code VARCHAR(50) PRIMARY KEY,
    inspection_type_name VARCHAR(100) NOT NULL,
    description TEXT,
    is_reactive BOOLEAN DEFAULT FALSE
);

-- Inspection Scopes lookup table
CREATE TABLE inspection_scopes (
    scope_code VARCHAR(50) PRIMARY KEY,
    scope_name VARCHAR(100) NOT NULL,
    description TEXT
);

-- Violation Types lookup table
CREATE TABLE violation_types (
    violation_type_code VARCHAR(50) PRIMARY KEY,
    violation_type_name VARCHAR(100) NOT NULL,
    severity_level INTEGER,
    description TEXT
);

-- Injury Degrees lookup table
CREATE TABLE injury_degrees (
    degree_code VARCHAR(20) PRIMARY KEY,
    degree_name VARCHAR(100) NOT NULL,
    severity_score INTEGER,
    description TEXT
);

-- Main entity tables

-- Create the database schema for OSHA safety analytics
CREATE TABLE IF NOT EXISTS establishments (
    establishment_id SERIAL PRIMARY KEY,
    establishment_name VARCHAR(255),
    address_city VARCHAR(100),
    address_state VARCHAR(2),
    naics_code VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS inspections (
    inspection_id INTEGER PRIMARY KEY,
    establishment_id INTEGER REFERENCES establishments(establishment_id),
    open_date DATE,
    inspection_type VARCHAR(50),
    has_injury BOOLEAN DEFAULT FALSE,
    total_violations INTEGER DEFAULT 0,
    total_penalty DECIMAL(12,2) DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS violations (
    violation_id SERIAL PRIMARY KEY,
    inspection_id INTEGER REFERENCES inspections(inspection_id),
    standard_violated VARCHAR(100),
    violation_type VARCHAR(50),
    penalty_amount DECIMAL(10,2),
    is_serious BOOLEAN DEFAULT FALSE,
    is_repeat BOOLEAN DEFAULT FALSE,
    is_willful BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id SERIAL PRIMARY KEY,
    inspection_id INTEGER REFERENCES inspections(inspection_id),
    establishment_id INTEGER REFERENCES establishments(establishment_id),
    predicted_risk_score DECIMAL(5,4),
    predicted_risk_category VARCHAR(20),
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Accidents table (optional, for detailed accident data)
CREATE TABLE accidents (
    accident_id SERIAL PRIMARY KEY,
    inspection_id INTEGER REFERENCES inspections(inspection_id),
    event_date DATE,
    event_description TEXT,
    degree_of_injury_code VARCHAR(20) REFERENCES injury_degrees(degree_code),
    nature_of_injury_code VARCHAR(50),
    part_of_body_affected_code VARCHAR(50),
    source_of_injury_code VARCHAR(50),
    number_fatalities INTEGER DEFAULT 0,
    number_hospitalized INTEGER DEFAULT 0,
    number_injured INTEGER DEFAULT 0,
    equipment_involved VARCHAR(255),
    task_being_performed TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model Versions table for tracking ML model versions
CREATE TABLE model_versions (
    model_version_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,  -- 'ensemble', 'random_forest', 'xgboost'
    training_date TIMESTAMP NOT NULL,
    training_data_range_start DATE,
    training_data_range_end DATE,
    training_records_count INTEGER,
    key_hyperparameters JSONB,
    performance_metrics JSONB,
    feature_importance JSONB,
    model_file_path VARCHAR(255),
    is_active BOOLEAN DEFAULT FALSE,
    created_by VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Identified Violation Patterns table
CREATE TABLE identified_violation_patterns (
    pattern_id SERIAL PRIMARY KEY,
    model_version_id INTEGER REFERENCES model_versions(model_version_id),
    pattern_name VARCHAR(255) NOT NULL,
    pattern_description TEXT,
    associated_standards TEXT[],  -- Array of OSHA standards
    associated_violation_types VARCHAR(50)[],
    industry_codes VARCHAR(10)[],  -- NAICS codes where pattern is common
    risk_score DECIMAL(5,4),  -- How predictive this pattern is
    frequency_score DECIMAL(5,4),  -- How common this pattern is
    impact_score DECIMAL(5,4),  -- Severity of outcomes when pattern occurs
    contributing_factors JSONB,
    examples JSONB,  -- Example cases demonstrating this pattern
    discovery_date DATE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance monitoring table
CREATE TABLE model_performance_log (
    log_id SERIAL PRIMARY KEY,
    model_version_id INTEGER REFERENCES model_versions(model_version_id),
    evaluation_date DATE,
    evaluation_period_start DATE,
    evaluation_period_end DATE,
    accuracy DECIMAL(5,4),
    precision_high_risk DECIMAL(5,4),
    recall_high_risk DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_roc DECIMAL(5,4),
    total_predictions INTEGER,
    correct_predictions INTEGER,
    false_positives INTEGER,
    false_negatives INTEGER,
    data_drift_score DECIMAL(5,4),
    performance_degradation_flag BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance

-- Indexes on foreign keys
CREATE INDEX idx_establishments_naics ON establishments(naics_code);
CREATE INDEX idx_inspections_establishment ON inspections(establishment_id);
CREATE INDEX idx_inspections_type ON inspections(inspection_type_code);
CREATE INDEX idx_violations_inspection ON violations(inspection_id);
CREATE INDEX idx_violations_type ON violations(violation_type_code);
CREATE INDEX idx_accidents_inspection ON accidents(inspection_id);
CREATE INDEX idx_predictions_model_version ON model_predictions(model_version_id);
CREATE INDEX idx_predictions_inspection ON model_predictions(inspection_id);
CREATE INDEX idx_predictions_establishment ON model_predictions(establishment_id);
CREATE INDEX idx_patterns_model_version ON identified_violation_patterns(model_version_id);

-- Indexes on commonly queried fields
CREATE INDEX idx_inspections_open_date ON inspections(open_date);
CREATE INDEX idx_inspections_industry_code ON inspections(industry_code);
CREATE INDEX idx_violations_standard ON violations(standard_violated);
CREATE INDEX idx_violations_serious ON violations(is_serious);
CREATE INDEX idx_violations_repeat ON violations(is_repeat);
CREATE INDEX idx_predictions_risk_category ON model_predictions(predicted_risk_category);
CREATE INDEX idx_predictions_timestamp ON model_predictions(prediction_timestamp);
CREATE INDEX idx_establishments_state ON establishments(address_state);
CREATE INDEX idx_establishments_city ON establishments(address_city);

-- Composite indexes for common query patterns
CREATE INDEX idx_inspections_date_industry ON inspections(open_date, industry_code);
CREATE INDEX idx_violations_inspection_type ON violations(inspection_id, violation_type_code);
CREATE INDEX idx_predictions_risk_timestamp ON model_predictions(predicted_risk_category, prediction_timestamp);

-- Create triggers for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_establishments_updated_at BEFORE UPDATE ON establishments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_inspections_updated_at BEFORE UPDATE ON inspections
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert initial lookup data

-- Inspection Types
INSERT INTO inspection_types (inspection_type_code, inspection_type_name, description, is_reactive) VALUES
('PLANNED', 'Planned Inspection', 'Routine scheduled inspection', FALSE),
('COMPLAINT', 'Complaint', 'Inspection based on worker complaint', TRUE),
('ACCIDENT', 'Accident Investigation', 'Investigation following workplace accident', TRUE),
('REFERRAL', 'Referral', 'Inspection based on referral from another agency', TRUE),
('FOLLOWUP', 'Follow-up', 'Follow-up to previous inspection', FALSE),
('MONITORING', 'Monitoring', 'Monitoring inspection', FALSE);

-- Inspection Scopes
INSERT INTO inspection_scopes (scope_code, scope_name, description) VALUES
('COMPLETE', 'Complete', 'Comprehensive inspection of entire workplace'),
('PARTIAL', 'Partial', 'Inspection of specific areas or hazards'),
('RECORDS', 'Records Only', 'Review of records without physical inspection');

-- Violation Types
INSERT INTO violation_types (violation_type_code, violation_type_name, severity_level, description) VALUES
('SERIOUS', 'Serious', 3, 'Violation where substantial probability of serious physical harm exists'),
('WILLFUL', 'Willful', 4, 'Violation committed intentionally or with plain indifference'),
('REPEAT', 'Repeat', 3, 'Violation substantially similar to previous citation'),
('OTHER', 'Other-than-Serious', 1, 'Violation that would probably not cause serious harm'),
('UNCLASS', 'Unclassified', 2, 'Unclassified violation');

-- Injury Degrees
INSERT INTO injury_degrees (degree_code, degree_name, severity_score, description) VALUES
('FATAL', 'Fatality', 4, 'Worker death'),
('HOSP', 'Hospitalized', 3, 'Worker hospitalized overnight'),
('NONHOSP', 'Non-hospitalized', 2, 'Injury not requiring hospitalization'),
('NONE', 'No Injury', 0, 'No injury reported');

-- Create views for common queries

-- View for high-risk establishments
CREATE VIEW high_risk_establishments AS
SELECT 
    e.establishment_id,
    e.establishment_name,
    e.address_city,
    e.address_state,
    e.naics_code,
    nc.naics_title,
    AVG(mp.predicted_risk_score) as avg_risk_score,
    COUNT(mp.prediction_id) as prediction_count,
    MAX(mp.prediction_timestamp) as latest_prediction
FROM establishments e
JOIN model_predictions mp ON e.establishment_id = mp.establishment_id
JOIN naics_codes nc ON e.naics_code = nc.naics_code
WHERE mp.predicted_risk_category IN ('High', 'Critical')
GROUP BY e.establishment_id, e.establishment_name, e.address_city, e.address_state, e.naics_code, nc.naics_title
HAVING AVG(mp.predicted_risk_score) > 0.7;

-- View for violation pattern analysis
CREATE VIEW violation_pattern_summary AS
SELECT 
    v.standard_violated,
    vt.violation_type_name,
    COUNT(*) as violation_count,
    AVG(v.penalty_amount_final) as avg_penalty,
    COUNT(CASE WHEN i.has_injury THEN 1 END) as associated_injuries,
    COUNT(CASE WHEN i.has_fatality THEN 1 END) as associated_fatalities
FROM violations v
JOIN violation_types vt ON v.violation_type_code = vt.violation_type_code
JOIN inspections i ON v.inspection_id = i.inspection_id
GROUP BY v.standard_violated, vt.violation_type_name
ORDER BY violation_count DESC;

-- View for model performance tracking
CREATE VIEW model_performance_summary AS
SELECT 
    mv.model_name,
    mv.model_version_id,
    mv.training_date,
    mpl.accuracy,
    mpl.precision_high_risk,
    mpl.recall_high_risk,
    mpl.f1_score,
    mpl.auc_roc,
    mpl.evaluation_date
FROM model_versions mv
JOIN model_performance_log mpl ON mv.model_version_id = mpl.model_version_id
WHERE mv.is_active = TRUE
ORDER BY mpl.evaluation_date DESC; 