-- ============================================================================
-- DWDM Lab Project - Week 1: Mumbai BMC Civic Complaints Warehouse
-- Script: 03_validation.sql
-- Description: Data Warehouse Integrity, Key, and Referential Validation Queries
-- ============================================================================

USE bmc_complaint_dw;

-- ----------------------------------------------------------------------------
-- 1. ROW COUNTS VALIDATION
-- Compare record counts across Fact and Dimension tables
-- ----------------------------------------------------------------------------
SELECT 'fact_complaint' AS table_name, COUNT(*) AS row_count FROM fact_complaint
UNION ALL
SELECT 'dim_time' AS table_name, COUNT(*) AS row_count FROM dim_time
UNION ALL
SELECT 'dim_location' AS table_name, COUNT(*) AS row_count FROM dim_location
UNION ALL
SELECT 'dim_category' AS table_name, COUNT(*) AS row_count FROM dim_category
UNION ALL
SELECT 'dim_status' AS table_name, COUNT(*) AS row_count FROM dim_status;

-- ----------------------------------------------------------------------------
-- 2. PRIMARY KEY UNIQUENESS VALIDATION (Must return 0 for all checks)
-- ----------------------------------------------------------------------------
-- Check Fact PK uniqueness
SELECT complaint_key, COUNT(*) AS duplicate_count
FROM fact_complaint
GROUP BY complaint_key
HAVING COUNT(*) > 1;

-- Check Business Key uniqueness
SELECT complaint_id, COUNT(*) AS duplicate_count
FROM fact_complaint
GROUP BY complaint_id
HAVING COUNT(*) > 1;

-- Check Dimension PK uniqueness
SELECT time_key, COUNT(*) AS duplicate_count FROM dim_time GROUP BY time_key HAVING COUNT(*) > 1;
SELECT location_key, COUNT(*) AS duplicate_count FROM dim_location GROUP BY location_key HAVING COUNT(*) > 1;
SELECT category_key, COUNT(*) AS duplicate_count FROM dim_category GROUP BY category_key HAVING COUNT(*) > 1;
SELECT status_key, COUNT(*) AS duplicate_count FROM dim_status GROUP BY status_key HAVING COUNT(*) > 1;

-- ----------------------------------------------------------------------------
-- 3. FOREIGN KEY REFERENTIAL INTEGRITY VALIDATION (Orphan Records Check)
-- Must return 0 orphaned records for all checks
-- ----------------------------------------------------------------------------
-- Check for orphan time_key
SELECT COUNT(*) AS orphan_time_records
FROM fact_complaint f
LEFT JOIN dim_time t ON f.time_key = t.time_key
WHERE t.time_key IS NULL;

-- Check for orphan location_key
SELECT COUNT(*) AS orphan_location_records
FROM fact_complaint f
LEFT JOIN dim_location l ON f.location_key = l.location_key
WHERE l.location_key IS NULL;

-- Check for orphan category_key
SELECT COUNT(*) AS orphan_category_records
FROM fact_complaint f
LEFT JOIN dim_category c ON f.category_key = c.category_key
WHERE c.category_key IS NULL;

-- Check for orphan status_key
SELECT COUNT(*) AS orphan_status_records
FROM fact_complaint f
LEFT JOIN dim_status s ON f.status_key = s.status_key
WHERE s.status_key IS NULL;

-- ----------------------------------------------------------------------------
-- 4. NULL VALUE AUDIT ON CRITICAL KEYS AND MEASURES
-- Must return 0 null values across all dimensions and measures
-- ----------------------------------------------------------------------------
SELECT
    SUM(CASE WHEN complaint_id IS NULL THEN 1 ELSE 0 END) AS null_complaint_ids,
    SUM(CASE WHEN time_key IS NULL THEN 1 ELSE 0 END) AS null_time_keys,
    SUM(CASE WHEN location_key IS NULL THEN 1 ELSE 0 END) AS null_location_keys,
    SUM(CASE WHEN category_key IS NULL THEN 1 ELSE 0 END) AS null_category_keys,
    SUM(CASE WHEN status_key IS NULL THEN 1 ELSE 0 END) AS null_status_keys,
    SUM(CASE WHEN complaint_count IS NULL THEN 1 ELSE 0 END) AS null_complaint_counts,
    SUM(CASE WHEN resolution_days IS NULL THEN 1 ELSE 0 END) AS null_resolution_days,
    SUM(CASE WHEN estimated_cost_inr IS NULL THEN 1 ELSE 0 END) AS null_estimated_costs
FROM fact_complaint;

-- ----------------------------------------------------------------------------
-- 5. SAMPLE WAREHOUSE JOIN QUERY (Verifying Star Schema Interoperability)
-- ----------------------------------------------------------------------------
SELECT
    f.complaint_key,
    t.full_date,
    l.ward,
    l.ward_area,
    c.category_name,
    s.status_name,
    f.resolution_days,
    f.estimated_cost_inr
FROM fact_complaint f
JOIN dim_time t
    ON f.time_key = t.time_key
JOIN dim_location l
    ON f.location_key = l.location_key
JOIN dim_category c
    ON f.category_key = c.category_key
JOIN dim_status s
    ON f.status_key = s.status_key
ORDER BY f.complaint_key ASC
LIMIT 20;
