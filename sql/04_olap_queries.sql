-- ============================================================================
-- DWDM Lab Project - Week 1: Mumbai BMC Civic Complaints Warehouse
-- Script: 04_olap_queries.sql
-- Description: Five Core OLAP Operations (Roll-up, Drill-down, Slice, Dice, Pivot)
-- ============================================================================

USE bmc_complaint_dw;

-- ============================================================================
-- 1. OLAP OPERATION 1: ROLL-UP (Aggregation to higher-level dimension)
-- Purpose:
--   Aggregate complaint metrics from lower geographic grain (Ward) to higher
--   geographic grain (Zone) along the location hierarchy: Area -> Ward -> Zone.
-- Interpretation:
--   Provides administrative leadership a high-level summary of civic issues
--   across Mumbai's primary administrative zones (Western, City, Eastern).
-- ============================================================================
SELECT
    l.zone,
    COUNT(f.complaint_key) AS total_complaints,
    ROUND(AVG(f.resolution_days), 2) AS avg_resolution_days,
    ROUND(SUM(f.estimated_cost_inr), 2) AS total_cost_inr
FROM fact_complaint f
JOIN dim_location l
    ON f.location_key = l.location_key
GROUP BY l.zone
ORDER BY total_complaints DESC;


-- ============================================================================
-- 2. OLAP OPERATION 2: DRILL-DOWN (Navigation to finer-level granularity)
-- Purpose:
--   Decompose higher-level annual complaint aggregations down into quarterly
--   and monthly time buckets along the time hierarchy: Year -> Quarter -> Month.
-- Interpretation:
--   Reveals seasonal spikes within the year (particularly during monsoon
--   months June-September) that are obscured when viewing annual aggregates.
-- ============================================================================
SELECT
    t.year,
    t.quarter,
    t.month,
    t.month_name,
    SUM(f.complaint_count) AS total_complaints,
    ROUND(AVG(f.resolution_days), 2) AS avg_resolution_days
FROM fact_complaint f
JOIN dim_time t
    ON f.time_key = t.time_key
WHERE t.year = 2024
GROUP BY
    t.year,
    t.quarter,
    t.month,
    t.month_name
ORDER BY
    t.quarter,
    t.month;


-- ============================================================================
-- 3. OLAP OPERATION 3: SLICE (Filter on exactly one dimension value)
-- Purpose:
--   Isolate a single slice of the data cube by fixing Complaint Category to
--   'Pothole / Road Damage' and inspecting its distribution across BMC Wards.
-- Interpretation:
--   Enables the BMC Roads & Traffic Department to identify specific suburban
--   and city wards with acute road infrastructure maintenance demands.
-- ============================================================================
SELECT
    l.ward,
    l.ward_area,
    l.zone,
    SUM(f.complaint_count) AS pothole_complaints,
    ROUND(AVG(f.resolution_days), 2) AS avg_resolution_days
FROM fact_complaint f
JOIN dim_location l
    ON f.location_key = l.location_key
JOIN dim_category c
    ON f.category_key = c.category_key
WHERE c.category_name = 'Pothole / Road Damage'
GROUP BY
    l.ward,
    l.ward_area,
    l.zone
ORDER BY
    pothole_complaints DESC;


-- ============================================================================
-- 4. OLAP OPERATION 4: DICE (Multi-dimensional sub-cube filtering)
-- Purpose:
--   Extract a focused sub-cube by restricting multiple dimensions simultaneously:
--   - Category IN ('Pothole / Road Damage', 'Water Supply Disruption', 'Drainage Overflow / Flooding')
--   - Zone IN ('Western', 'Eastern')
--   - Year = 2024
-- Interpretation:
--   Allows comparative analysis of the three critical utility disruptions
--   between the Western and Eastern suburbs during the 2024 reporting period.
-- ============================================================================
SELECT
    t.year,
    l.zone,
    c.category_name,
    SUM(f.complaint_count) AS total_complaints,
    ROUND(AVG(f.resolution_days), 2) AS avg_resolution_days
FROM fact_complaint f
JOIN dim_location l
    ON f.location_key = l.location_key
JOIN dim_category c
    ON f.category_key = c.category_key
JOIN dim_time t
    ON f.time_key = t.time_key
WHERE c.category_name IN ('Pothole / Road Damage', 'Water Supply Disruption', 'Drainage Overflow / Flooding')
  AND l.zone IN ('Western', 'Eastern')
  AND t.year = 2024
GROUP BY
    t.year,
    l.zone,
    c.category_name
ORDER BY
    l.zone,
    total_complaints DESC;


-- ============================================================================
-- 5. OLAP OPERATION 5: PIVOT (Dimension rotation / Cross-tabulation)
-- Purpose:
--   Rotate the Category dimension against the Location (Ward) dimension using
--   conditional aggregation, converting category rows into dedicated metric columns.
-- Interpretation:
--   Produces an executive cross-tabulation matrix showing ward-by-ward
--   breakdown across the top civic issue types for resource allocation.
-- ============================================================================
SELECT
    l.ward,
    l.ward_area,
    SUM(CASE WHEN c.category_name = 'Pothole / Road Damage' THEN f.complaint_count ELSE 0 END) AS pothole_roads,
    SUM(CASE WHEN c.category_name = 'Water Supply Disruption' THEN f.complaint_count ELSE 0 END) AS water_supply,
    SUM(CASE WHEN c.category_name = 'Solid Waste / Garbage' THEN f.complaint_count ELSE 0 END) AS solid_waste,
    SUM(CASE WHEN c.category_name = 'Drainage Overflow / Flooding' THEN f.complaint_count ELSE 0 END) AS drainage_flooding,
    SUM(f.complaint_count) AS total_ward_complaints
FROM fact_complaint f
JOIN dim_location l
    ON f.location_key = l.location_key
JOIN dim_category c
    ON f.category_key = c.category_key
GROUP BY
    l.ward,
    l.ward_area
ORDER BY
    total_ward_complaints DESC;
