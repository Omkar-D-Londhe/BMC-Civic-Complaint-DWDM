"""
DWDM Project: Mumbai BMC Civic Grievance Recurrence Analysis
Analyzes repeat complainant behavior, chronic grievance hotspots, and prior complaint distributions:
- Repeat complaint rates by Ward and Category
- Prior complaints distribution (0, 1, 2, 3, 4, 5+)
- Satisfaction vs Recurrence dynamics
- Outputs: outputs/mining/recurrence_patterns.json
"""

import json
import os
from dotenv import load_dotenv
import pandas as pd
import pymysql

load_dotenv()


def get_db_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "bmc_complaint_dw"),
        port=int(os.getenv("DB_PORT", 3306)),
    )


def analyze_recurrence():
    conn = get_db_connection()
    print("Extracting recurrence telemetry from fact_complaint...")

    # Overall KPIs
    kpi_query = """
    SELECT 
        COUNT(*) AS total_complaints,
        SUM(repeat_complainant) AS repeat_complaints_count,
        AVG(repeat_complainant) * 100 AS repeat_complaint_rate,
        AVG(prior_complaints_count) AS avg_prior_complaints,
        AVG(CASE WHEN repeat_complainant = 1 THEN citizen_satisfied ELSE NULL END) * 100 AS repeat_satisfaction_rate,
        AVG(CASE WHEN repeat_complainant = 0 THEN citizen_satisfied ELSE NULL END) * 100 AS first_time_satisfaction_rate,
        AVG(CASE WHEN repeat_complainant = 1 THEN resolution_days ELSE NULL END) AS repeat_avg_resolution_days,
        AVG(CASE WHEN repeat_complainant = 0 THEN resolution_days ELSE NULL END) AS first_time_avg_resolution_days
    FROM fact_complaint;
    """
    kpis = pd.read_sql(kpi_query, conn).iloc[0].to_dict()

    # Repeat rate by category
    cat_query = """
    SELECT 
        c.category_name,
        COUNT(f.complaint_key) AS total,
        SUM(f.repeat_complainant) AS repeat_count,
        ROUND(AVG(f.repeat_complainant) * 100, 2) AS repeat_pct,
        ROUND(AVG(f.prior_complaints_count), 2) AS avg_prior_complaints
    FROM fact_complaint f
    JOIN dim_category c ON f.category_key = c.category_key
    GROUP BY c.category_name
    ORDER BY repeat_pct DESC;
    """
    cat_df = pd.read_sql(cat_query, conn)

    # Repeat rate by ward
    ward_query = """
    SELECT 
        l.ward,
        l.ward_area,
        l.zone,
        COUNT(f.complaint_key) AS total,
        SUM(f.repeat_complainant) AS repeat_count,
        ROUND(AVG(f.repeat_complainant) * 100, 2) AS repeat_pct
    FROM fact_complaint f
    JOIN dim_location l ON f.location_key = l.location_key
    GROUP BY l.ward, l.ward_area, l.zone
    ORDER BY repeat_pct DESC;
    """
    ward_df = pd.read_sql(ward_query, conn)

    # Prior complaints frequency distribution
    dist_query = """
    SELECT 
        CASE 
            WHEN prior_complaints_count = 0 THEN '0 prior'
            WHEN prior_complaints_count = 1 THEN '1 prior'
            WHEN prior_complaints_count = 2 THEN '2 prior'
            WHEN prior_complaints_count = 3 THEN '3 prior'
            WHEN prior_complaints_count = 4 THEN '4 prior'
            ELSE '5+ prior'
        END AS prior_bucket,
        COUNT(*) AS count,
        ROUND(AVG(citizen_satisfied) * 100, 2) AS satisfaction_rate
    FROM fact_complaint
    GROUP BY prior_bucket
    ORDER BY count DESC;
    """
    dist_df = pd.read_sql(dist_query, conn)
    conn.close()

    result = {
        "kpis": {
            "total_complaints": int(kpis["total_complaints"]),
            "repeat_complaints_count": int(kpis["repeat_complaints_count"]),
            "repeat_complaint_rate": round(float(kpis["repeat_complaint_rate"]), 2),
            "avg_prior_complaints": round(float(kpis["avg_prior_complaints"]), 2),
            "repeat_satisfaction_rate": round(float(kpis["repeat_satisfaction_rate"]), 2),
            "first_time_satisfaction_rate": round(float(kpis["first_time_satisfaction_rate"]), 2),
            "repeat_avg_resolution_days": round(float(kpis["repeat_avg_resolution_days"]), 1),
            "first_time_avg_resolution_days": round(float(kpis["first_time_avg_resolution_days"]), 1),
        },
        "by_category": cat_df.to_dict(orient="records"),
        "by_ward": ward_df.to_dict(orient="records"),
        "prior_distribution": dist_df.to_dict(orient="records"),
    }

    os.makedirs(os.path.join("outputs", "mining"), exist_ok=True)
    out_file = os.path.join("outputs", "mining", "recurrence_patterns.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Recurrence analysis completed. Output saved to {out_file}")


if __name__ == "__main__":
    analyze_recurrence()
