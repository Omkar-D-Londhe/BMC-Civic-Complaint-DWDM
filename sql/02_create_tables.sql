-- ============================================================================
-- DWDM Lab Project - Week 1: Mumbai BMC Civic Complaints Warehouse
-- Script: 02_create_tables.sql
-- Description: Create Star Schema dimension tables and fact table in MySQL
-- ============================================================================

USE bmc_complaint_dw;

-- Disable foreign key checks for clean re-creation
SET FOREIGN_KEY_CHECKS = 0;

-- Drop existing tables if they exist
DROP TABLE IF EXISTS fact_complaint;
DROP TABLE IF EXISTS dim_status;
DROP TABLE IF EXISTS dim_category;
DROP TABLE IF EXISTS dim_location;
DROP TABLE IF EXISTS dim_time;

-- ----------------------------------------------------------------------------
-- 1. DIMENSION: DIM_TIME
-- Hierarchies: Day -> Month -> Quarter -> Year
-- ----------------------------------------------------------------------------
CREATE TABLE dim_time (
    time_key INT PRIMARY KEY,
    full_date DATE NOT NULL UNIQUE,
    day INT NOT NULL,
    month INT NOT NULL,
    month_name VARCHAR(15) NOT NULL,
    quarter INT NOT NULL,
    year INT NOT NULL,
    day_of_week VARCHAR(15) NOT NULL,
    is_monsoon_season INT NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ----------------------------------------------------------------------------
-- 2. DIMENSION: DIM_LOCATION
-- Hierarchies: Area (ward_area) -> Ward (ward) -> Zone
-- ----------------------------------------------------------------------------
CREATE TABLE dim_location (
    location_key INT PRIMARY KEY,
    ward VARCHAR(10) NOT NULL UNIQUE,
    ward_area VARCHAR(100) NOT NULL,
    zone VARCHAR(50) NOT NULL,
    ward_type VARCHAR(50) NOT NULL,
    population_density VARCHAR(50) NOT NULL,
    ward_slum_percentage DECIMAL(5,2) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ----------------------------------------------------------------------------
-- 3. DIMENSION: DIM_CATEGORY
-- Hierarchies: Complaint Category -> Responsible Department
-- ----------------------------------------------------------------------------
CREATE TABLE dim_category (
    category_key INT PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL UNIQUE,
    department_assigned VARCHAR(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ----------------------------------------------------------------------------
-- 4. DIMENSION: DIM_STATUS
-- Captures complaint resolution status, work quality, and contractor category
-- ----------------------------------------------------------------------------
CREATE TABLE dim_status (
    status_key INT PRIMARY KEY,
    status_name VARCHAR(50) NOT NULL,
    work_quality_rating VARCHAR(50) NOT NULL,
    contractor_category VARCHAR(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ----------------------------------------------------------------------------
-- 5. FACT TABLE: FACT_COMPLAINT
-- Grain: One row represents one unique civic complaint registered with BMC.
-- Measures:
--   - complaint_count (Fully Additive)
--   - resolution_days (Additive / Semi-Additive)
--   - estimated_cost_inr (Fully Additive)
--   - num_reassignments (Fully Additive)
--   - prior_complaints_count (Fully Additive)
--   - infrastructure_age_years (Semi-Additive / Non-Additive)
--   - months_since_last_maintained (Semi-Additive / Non-Additive)
--   - citizen_satisfied (Additive 0/1 binary satisfaction indicator)
-- ----------------------------------------------------------------------------
CREATE TABLE fact_complaint (
    complaint_key INT AUTO_INCREMENT PRIMARY KEY,
    complaint_id VARCHAR(50) NOT NULL UNIQUE,
    time_key INT NOT NULL,
    location_key INT NOT NULL,
    category_key INT NOT NULL,
    status_key INT NOT NULL,
    complaint_channel VARCHAR(50) NOT NULL,
    complaint_time_of_day VARCHAR(50) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    complainant_type VARCHAR(50) NOT NULL,
    property_type VARCHAR(50) NOT NULL,
    complaint_count INT NOT NULL DEFAULT 1,
    resolution_days INT NOT NULL,
    estimated_cost_inr DECIMAL(12,2) NOT NULL,
    num_reassignments INT NOT NULL,
    prior_complaints_count INT NOT NULL,
    infrastructure_age_years INT NOT NULL,
    months_since_last_maintained INT NOT NULL,
    citizen_satisfied INT NOT NULL,
    has_photo_evidence INT NOT NULL,
    has_gps_location INT NOT NULL,
    media_attention INT NOT NULL,
    politically_sensitive INT NOT NULL,
    repeat_complainant INT NOT NULL,
    site_inspected INT NOT NULL,
    defect_liability_claim INT NOT NULL,
    CONSTRAINT fk_fact_time FOREIGN KEY (time_key) REFERENCES dim_time(time_key),
    CONSTRAINT fk_fact_location FOREIGN KEY (location_key) REFERENCES dim_location(location_key),
    CONSTRAINT fk_fact_category FOREIGN KEY (category_key) REFERENCES dim_category(category_key),
    CONSTRAINT fk_fact_status FOREIGN KEY (status_key) REFERENCES dim_status(status_key)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Performance indexes on foreign keys
CREATE INDEX idx_fact_time ON fact_complaint(time_key);
CREATE INDEX idx_fact_location ON fact_complaint(location_key);
CREATE INDEX idx_fact_category ON fact_complaint(category_key);
CREATE INDEX idx_fact_status ON fact_complaint(status_key);

-- Re-enable foreign key checks
SET FOREIGN_KEY_CHECKS = 1;
