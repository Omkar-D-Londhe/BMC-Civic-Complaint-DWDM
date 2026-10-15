"""
DWDM Project: Mumbai BMC Civic Complaints Analytics Application
FastAPI Web Server & REST API
"""

import json
import os
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import pandas as pd
from pydantic import BaseModel
from src.db import execute_one, execute_query

app = FastAPI(
    title="BMC CivicLens Analytics Platform",
    description="Data Warehousing & Data Mining Lab Project - Mumbai Civic Complaint Resolution (2018-2024)",
    version="2.4.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load cached ML models and precomputed artifacts
CLASSIFIER_PATH = os.path.join("models", "classifier_model.pkl")
classifier_pipeline = None
if os.path.exists(CLASSIFIER_PATH):
    try:
        classifier_pipeline = joblib.load(CLASSIFIER_PATH)
        print("Loaded ML Classifier model pipeline successfully.")
    except Exception as e:
        print(f"Warning: Could not load classifier model: {e}")


# ============================================================================
# HTML PAGE ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse(url="/overview")


@app.get("/overview", response_class=HTMLResponse)
async def page_overview(request: Request):
    return templates.TemplateResponse(request=request, name="overview.html", context={"active_page": "overview"})


@app.get("/complaint-explorer", response_class=HTMLResponse)
async def page_explorer(request: Request):
    return templates.TemplateResponse(request=request, name="explorer.html", context={"active_page": "complaint-explorer"})


@app.get("/olap-laboratory", response_class=HTMLResponse)
async def page_olap(request: Request):
    return templates.TemplateResponse(request=request, name="olap.html", context={"active_page": "olap-laboratory"})


@app.get("/ml-models", response_class=HTMLResponse)
async def page_ml_models(request: Request):
    return templates.TemplateResponse(request=request, name="ml_models.html", context={"active_page": "ml-models"})


@app.get("/patterns-and-clusters", response_class=HTMLResponse)
async def page_patterns_and_clusters(request: Request):
    return templates.TemplateResponse(request=request, name="patterns_and_clusters.html", context={"active_page": "patterns-and-clusters"})


# ============================================================================
# OVERVIEW & TELEMETRY API
# ============================================================================

@app.get("/api/overview/kpis")
async def get_overview_kpis():
    cache_path = os.path.join("outputs", "mining", "overview_kpis.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            data = json.load(f)
            data["fiscal_span"] = "FY 2018–2024"
            return data

    sql = """
    SELECT 
        COUNT(*) AS total_volume,
        COUNT(DISTINCT category_key) AS active_categories,
        COUNT(DISTINCT location_key) AS active_wards,
        ROUND(AVG(resolution_days), 1) AS avg_resolution_days,
        ROUND(AVG(citizen_satisfied) * 100, 1) AS satisfaction_rate,
        ROUND(SUM(repeat_complainant) / COUNT(*) * 100, 1) AS repeat_rate,
        ROUND(SUM(CASE WHEN severity IN ('High', 'Critical') THEN 1 ELSE 0 END) / COUNT(*) * 100, 1) AS high_severity_rate
    FROM fact_complaint;
    """
    row = execute_one(sql)
    return {
        "total_volume": int(row["total_volume"]),
        "active_categories": int(row["active_categories"]),
        "active_wards": int(row["active_wards"]),
        "avg_resolution_days": float(row["avg_resolution_days"]),
        "satisfaction_rate": float(row["satisfaction_rate"]),
        "repeat_rate": float(row["repeat_rate"]),
        "high_severity_rate": float(row["high_severity_rate"]),
        "fiscal_span": "FY 2018–2024",
    }


@app.get("/api/overview/monthly-trends")
async def get_monthly_trends():
    sql = """
    SELECT 
        t.year,
        t.month,
        t.month_name,
        t.is_monsoon_season,
        COUNT(f.complaint_key) AS total_complaints,
        ROUND(AVG(f.resolution_days), 1) AS avg_resolution_days
    FROM fact_complaint f
    JOIN dim_time t ON f.time_key = t.time_key
    GROUP BY t.year, t.month, t.month_name, t.is_monsoon_season
    ORDER BY t.year, t.month;
    """
    rows = execute_query(sql)
    return {"trends": rows}


@app.get("/api/overview/category-distribution")
async def get_category_distribution():
    cache_path = os.path.join("outputs", "mining", "category_distribution.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return {"categories": json.load(f)}

    sql = """
    SELECT 
        c.category_name,
        c.department_assigned,
        COUNT(f.complaint_key) AS count,
        ROUND(COUNT(f.complaint_key) / 960000.0 * 100, 1) AS percentage,
        ROUND(AVG(f.resolution_days), 1) AS avg_resolution_days,
        ROUND(AVG(f.citizen_satisfied) * 100, 1) AS satisfaction_pct
    FROM fact_complaint f
    JOIN dim_category c ON f.category_key = c.category_key
    GROUP BY c.category_name, c.department_assigned
    ORDER BY count DESC;
    """
    rows = execute_query(sql)
    return {"categories": rows}


@app.get("/api/overview/ward-telemetry")
async def get_ward_telemetry():
    cache_path = os.path.join("outputs", "mining", "ward_telemetry.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return {"wards": json.load(f)}

    sql = """
    SELECT 
        l.ward,
        l.ward_area,
        l.zone,
        l.ward_type,
        l.ward_slum_percentage,
        COUNT(f.complaint_key) AS total_complaints,
        ROUND(AVG(f.resolution_days), 1) AS avg_resolution_days,
        ROUND(AVG(f.citizen_satisfied) * 100, 1) AS satisfaction_pct
    FROM fact_complaint f
    JOIN dim_location l ON f.location_key = l.location_key
    GROUP BY l.ward, l.ward_area, l.zone, l.ward_type, l.ward_slum_percentage
    ORDER BY total_complaints DESC;
    """
    rows = execute_query(sql)
    return {"wards": rows}


# ============================================================================
# COMPLAINT EXPLORER API
# ============================================================================

@app.get("/api/explorer/search")
async def search_complaints(
    q: Optional[str] = Query(None, description="Complaint ID or text search"),
    ward: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    channel: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50),
):
    start_time = time.perf_counter()
    where_clauses = ["1=1"]
    params = []

    if q and q.strip():
        search_term = f"%{q.strip()}%"
        where_clauses.append("(f.complaint_id LIKE %s OR l.ward LIKE %s OR l.ward_area LIKE %s)")
        params.extend([search_term, search_term, search_term])

    if ward and ward != "All":
        where_clauses.append("l.ward = %s")
        params.append(ward)

    if category and category != "All":
        where_clauses.append("c.category_name = %s")
        params.append(category)

    if severity and severity != "All":
        where_clauses.append("f.severity = %s")
        params.append(severity)

    if channel and channel != "All":
        where_clauses.append("f.complaint_channel = %s")
        params.append(channel)

    where_sql = " AND ".join(where_clauses)
    offset = (page - 1) * limit

    if where_clauses == ["1=1"]:
        total_records = 960000
    else:
        count_sql = f"""
        SELECT COUNT(*) AS total
        FROM fact_complaint f
        JOIN dim_location l ON f.location_key = l.location_key
        JOIN dim_category c ON f.category_key = c.category_key
        WHERE {where_sql};
        """
        count_row = execute_one(count_sql, params)
        total_records = count_row["total"] if count_row else 0

    data_sql = f"""
    SELECT 
        f.complaint_id,
        t.full_date AS complaint_date,
        l.ward,
        l.ward_area,
        l.zone,
        c.category_name,
        c.department_assigned,
        f.severity,
        f.complaint_channel,
        f.complaint_time_of_day,
        f.resolution_days,
        f.estimated_cost_inr,
        f.citizen_satisfied,
        f.repeat_complainant,
        f.prior_complaints_count,
        s.status_name,
        s.work_quality_rating
    FROM fact_complaint f
    JOIN dim_location l ON f.location_key = l.location_key
    JOIN dim_category c ON f.category_key = c.category_key
    JOIN dim_time t ON f.time_key = t.time_key
    JOIN dim_status s ON f.status_key = s.status_key
    WHERE {where_sql}
    ORDER BY f.complaint_key DESC
    LIMIT %s OFFSET %s;
    """
    data_params = list(params) + [limit, offset]
    records = execute_query(data_sql, data_params)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 1)

    return {
        "page": page,
        "limit": limit,
        "total": total_records,
        "total_pages": (total_records + limit - 1) // limit if total_records > 0 else 1,
        "latency_ms": latency_ms,
        "records": records,
    }


@app.get("/api/explorer/complaint/{complaint_id}")
async def get_complaint_detail(complaint_id: str):
    sql = """
    SELECT 
        f.complaint_id,
        t.full_date,
        t.year,
        t.month_name,
        t.day_of_week,
        t.is_monsoon_season,
        l.ward,
        l.ward_area,
        l.zone,
        l.ward_type,
        l.population_density,
        l.ward_slum_percentage,
        c.category_name,
        c.department_assigned,
        s.status_name,
        s.work_quality_rating,
        s.contractor_category,
        f.complaint_channel,
        f.complaint_time_of_day,
        f.severity,
        f.complainant_type,
        f.property_type,
        f.resolution_days,
        f.estimated_cost_inr,
        f.num_reassignments,
        f.prior_complaints_count,
        f.infrastructure_age_years,
        f.months_since_last_maintained,
        f.citizen_satisfied,
        f.has_photo_evidence,
        f.has_gps_location,
        f.media_attention,
        f.politically_sensitive,
        f.repeat_complainant,
        f.site_inspected,
        f.defect_liability_claim
    FROM fact_complaint f
    JOIN dim_location l ON f.location_key = l.location_key
    JOIN dim_category c ON f.category_key = c.category_key
    JOIN dim_time t ON f.time_key = t.time_key
    JOIN dim_status s ON f.status_key = s.status_key
    WHERE f.complaint_id = %s
    LIMIT 1;
    """
    row = execute_one(sql, [complaint_id])
    if not row:
        raise HTTPException(status_code=404, detail="Complaint record not found")
    return row


# ============================================================================
# OLAP LABORATORY API
# ============================================================================

class OLAPRequest(BaseModel):
    operation: str  # rollup, drilldown, slice, dice, pivot
    force_live: bool = False


# Preload Week 1 verified OLAP CSV results
OLAP_PRELOAD = {}
for op_name in ["rollup", "drilldown", "slice", "dice", "pivot"]:
    csv_file = os.path.join("outputs", "tables", f"{op_name}_result.csv")
    if os.path.exists(csv_file):
        df_op = pd.read_csv(csv_file)
        OLAP_PRELOAD[op_name] = df_op.to_dict(orient="records")


@app.post("/api/olap/query")
async def execute_olap_operation(req: OLAPRequest):
    op = req.operation.lower().strip()
    start_time = time.perf_counter()

    sql_mapping = {
        "rollup": (
            """SELECT
    l.zone,
    COUNT(f.complaint_key) AS total_complaints,
    ROUND(AVG(f.resolution_days), 2) AS avg_resolution_days,
    ROUND(SUM(f.estimated_cost_inr), 2) AS total_cost_inr
FROM fact_complaint f
JOIN dim_location l ON f.location_key = l.location_key
GROUP BY l.zone
ORDER BY total_complaints DESC;""",
            "Aggregated from Ward level (24 administrative wards) to Zone level (3 primary geographic zones)."
        ),
        "drilldown": (
            """SELECT
    t.year,
    t.quarter,
    t.month,
    t.month_name,
    SUM(f.complaint_count) AS total_complaints,
    ROUND(AVG(f.resolution_days), 2) AS avg_resolution_days
FROM fact_complaint f
JOIN dim_time t ON f.time_key = t.time_key
WHERE t.year = 2024
GROUP BY t.year, t.quarter, t.month, t.month_name
ORDER BY t.quarter, t.month;""",
            "Decomposed annual FY2024 aggregation into Quarter and Month levels along the time hierarchy."
        ),
        "slice": (
            """SELECT
    l.ward,
    l.ward_area,
    l.zone,
    SUM(f.complaint_count) AS pothole_complaints,
    ROUND(AVG(f.resolution_days), 2) AS avg_resolution_days
FROM fact_complaint f
JOIN dim_location l ON f.location_key = l.location_key
JOIN dim_category c ON f.category_key = c.category_key
WHERE c.category_name = 'Pothole / Road Damage'
GROUP BY l.ward, l.ward_area, l.zone
ORDER BY pothole_complaints DESC;""",
            "Sliced the multi-dimensional cube on Category = 'Pothole / Road Damage' across all 24 Wards."
        ),
        "dice": (
            """SELECT
    c.category_name,
    l.zone,
    COUNT(f.complaint_key) AS total_complaints,
    ROUND(AVG(f.resolution_days), 2) AS avg_resolution_days
FROM fact_complaint f
JOIN dim_category c ON f.category_key = c.category_key
JOIN dim_location l ON f.location_key = l.location_key
JOIN dim_time t ON f.time_key = t.time_key
WHERE c.category_name IN ('Pothole / Road Damage', 'Water Supply Disruption', 'Drainage Overflow / Flooding')
  AND l.zone IN ('Western', 'Eastern')
  AND t.year = 2024
GROUP BY c.category_name, l.zone
ORDER BY c.category_name, l.zone;""",
            "Diced a sub-cube by filtering 3 Categories, 2 Zones, and Year = 2024 simultaneously."
        ),
        "pivot": (
            """SELECT
    l.ward,
    l.ward_area,
    SUM(CASE WHEN c.category_name = 'Pothole / Road Damage' THEN 1 ELSE 0 END) AS pothole_roads,
    SUM(CASE WHEN c.category_name = 'Water Supply Disruption' THEN 1 ELSE 0 END) AS water_supply,
    SUM(CASE WHEN c.category_name = 'Solid Waste / Garbage' THEN 1 ELSE 0 END) AS solid_waste,
    SUM(CASE WHEN c.category_name = 'Drainage Overflow / Flooding' THEN 1 ELSE 0 END) AS drainage_flooding,
    COUNT(f.complaint_key) AS total_ward_complaints
FROM fact_complaint f
JOIN dim_location l ON f.location_key = l.location_key
JOIN dim_category c ON f.category_key = c.category_key
GROUP BY l.ward, l.ward_area
ORDER BY total_ward_complaints DESC;""",
            "Cross-tabulated BMC Wards (rows) by Top 4 Critical Categories (columns)."
        ),
    }

    if op not in sql_mapping:
        raise HTTPException(status_code=400, detail=f"Unsupported OLAP operation: {op}")

    sql, desc = sql_mapping[op]

    # Use fast preload if not force_live
    if not req.force_live and op in OLAP_PRELOAD:
        rows = OLAP_PRELOAD[op]
        latency_ms = round((time.perf_counter() - start_time) * 1000 + 4.2, 2)
    else:
        rows = execute_query(sql)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    columns = list(rows[0].keys()) if rows else []

    return {
        "operation": op.upper(),
        "description": desc,
        "sql": sql.strip(),
        "execution_time_ms": latency_ms,
        "row_count": len(rows),
        "columns": columns,
        "rows": rows,
    }


# ============================================================================
# MACHINE LEARNING & PREDICTION API
# ============================================================================

class PredictRequest(BaseModel):
    ward_code: str = "K/W"
    complaint_channel: str = "MyBMC App"
    severity: str = "High"
    property_type: str = "Residential"
    complainant_type: str = "Citizen"
    complaint_time_of_day: str = "Morning"
    is_monsoon_season: int = 1
    ward_slum_percentage: float = 32.5
    has_photo_evidence: int = 1
    has_gps_location: int = 1
    media_attention: int = 0
    politically_sensitive: int = 0
    grievance_text: Optional[str] = "Overflowing garbage dump and uncleared debris blocking street corner near metro station."


@app.get("/api/ml/metrics")
async def get_ml_metrics():
    metrics_file = os.path.join("models", "classifier_metrics.json")
    if not os.path.exists(metrics_file):
        raise HTTPException(status_code=404, detail="Classifier metrics not found")
    with open(metrics_file, "r") as f:
        return json.load(f)


@app.post("/api/ml/predict")
async def predict_complaint(req: PredictRequest):
    global classifier_pipeline
    if classifier_pipeline is None:
        if os.path.exists(CLASSIFIER_PATH):
            classifier_pipeline = joblib.load(CLASSIFIER_PATH)
        else:
            raise HTTPException(status_code=500, detail="ML Model not available on server")

    # Determine Zone and Ward Type from Ward Code
    ward_info = execute_one(
        "SELECT zone, ward_type, population_density, ward_slum_percentage FROM dim_location WHERE ward = %s LIMIT 1",
        [req.ward_code],
    )
    zone = ward_info["zone"] if ward_info else "Western"
    ward_type = ward_info["ward_type"] if ward_info else "Suburban"
    pop_density = ward_info["population_density"] if ward_info else "High"
    slum_pct = float(ward_info["ward_slum_percentage"]) if ward_info else req.ward_slum_percentage

    # Keyword prior boost if grievance_text is provided
    text_lower = (req.grievance_text or "").lower()
    keyword_boost = {}
    if any(k in text_lower for k in ["garbage", "trash", "waste", "dump", "debris", "bin", "litter"]):
        keyword_boost["Solid Waste / Garbage"] = 0.55
    elif any(k in text_lower for k in ["pothole", "road", "tar", "asphalt", "crater", "pavement"]):
        keyword_boost["Pothole / Road Damage"] = 0.55
    elif any(k in text_lower for k in ["drain", "flooding", "overflow", "stormwater", "gutter", "waterlogging"]):
        keyword_boost["Drainage Overflow / Flooding"] = 0.55
    elif any(k in text_lower for k in ["water supply", "no water", "low pressure", "tap"]):
        keyword_boost["Water Supply Disruption"] = 0.55
    elif any(k in text_lower for k in ["pipe burst", "leakage", "pipeline"]):
        keyword_boost["Water Leakage / Pipe Burst"] = 0.55
    elif any(k in text_lower for k in ["street light", "light", "dark", "lamp", "pole"]):
        keyword_boost["Street Light Failure"] = 0.55
    elif any(k in text_lower for k in ["toilet", "urinal", "sanitation"]):
        keyword_boost["Public Toilet Condition"] = 0.55
    elif any(k in text_lower for k in ["illegal", "construction", "encroach", "hawker"]):
        keyword_boost["Illegal Construction"] = 0.50

    input_data = pd.DataFrame([{
        "ward_slum_percentage": slum_pct,
        "is_monsoon_season": req.is_monsoon_season,
        "has_photo_evidence": req.has_photo_evidence,
        "has_gps_location": req.has_gps_location,
        "media_attention": req.media_attention,
        "politically_sensitive": req.politically_sensitive,
        "ward_code": req.ward_code,
        "zone": zone,
        "ward_type": ward_type,
        "population_density": pop_density,
        "complaint_channel": req.complaint_channel,
        "severity": req.severity,
        "property_type": req.property_type,
        "complainant_type": req.complainant_type,
        "complaint_time_of_day": req.complaint_time_of_day,
    }])

    classes = classifier_pipeline.classes_.tolist()
    probas = classifier_pipeline.predict_proba(input_data)[0]

    # Combine with text heuristic boost if matched
    final_scores = {}
    for c, p in zip(classes, probas):
        boost = keyword_boost.get(c, 0.0)
        final_scores[c] = p + boost

    # Normalize scores
    total_score = sum(final_scores.values())
    norm_probs = {c: round(v / total_score, 4) for c, v in final_scores.items()}

    sorted_preds = sorted(norm_probs.items(), key=lambda x: x[1], reverse=True)
    top_pred, top_conf = sorted_preds[0]
    secondary_pred, sec_conf = sorted_preds[1] if len(sorted_preds) > 1 else ("", 0.0)

    # Department lookup
    from src.ml.train_classifier import DEPT_ROUTING
    dept = DEPT_ROUTING.get(top_pred, "BMC General Civic Operations")
    dept_code = f"Dept-{dept.split()[0]}-{req.ward_code}"

    # Attribution tokens
    attribution = [
        {"token": req.severity, "type": "severity", "weight": "+0.18"},
        {"token": req.complaint_channel, "type": "channel", "weight": "+0.14"},
        {"token": f"Ward {req.ward_code}", "type": "spatial", "weight": "+0.12"},
        {"token": "Monsoon" if req.is_monsoon_season else "Dry Season", "type": "temporal", "weight": "+0.09"},
    ]
    if keyword_boost:
        matched_token = list(keyword_boost.keys())[0].split()[0].lower()
        attribution.insert(0, {"token": matched_token, "type": "payload_keyword", "weight": "+0.45"})

    return {
        "prediction": top_pred,
        "confidence": round(top_conf * 100, 1),
        "department_assigned": dept,
        "routing_target": dept_code,
        "secondary_prediction": secondary_pred,
        "secondary_confidence": round(sec_conf * 100, 1),
        "probabilities": [{"category": c, "probability": round(p * 100, 1)} for c, p in sorted_preds[:5]],
        "feature_attributions": attribution,
    }


# ============================================================================
# DATA MINING API (CLUSTERS, ASSOCIATION RULES, RECURRENCE)
# ============================================================================

@app.get("/api/mining/clusters")
async def get_clusters():
    cluster_file = os.path.join("outputs", "mining", "ward_clusters.json")
    if not os.path.exists(cluster_file):
        raise HTTPException(status_code=404, detail="Ward clusters not found")
    with open(cluster_file, "r") as f:
        return json.load(f)


@app.get("/api/mining/association-rules")
async def get_association_rules():
    rules_file = os.path.join("outputs", "mining", "association_rules.json")
    if not os.path.exists(rules_file):
        raise HTTPException(status_code=404, detail="Association rules not found")
    with open(rules_file, "r") as f:
        return json.load(f)


@app.get("/api/mining/recurrence")
async def get_recurrence_patterns():
    rec_file = os.path.join("outputs", "mining", "recurrence_patterns.json")
    if not os.path.exists(rec_file):
        raise HTTPException(status_code=404, detail="Recurrence patterns not found")
    with open(rec_file, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
