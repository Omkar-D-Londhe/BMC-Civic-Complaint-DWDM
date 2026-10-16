"""
Comprehensive Verification Test Suite for BMC CivicLens Analytics Platform
Tests all HTML pages, REST API endpoints, ML inference pipelines, OLAP queries, and data mining models.
"""

import time
import json
from app import (
    app,
    get_overview_kpis,
    get_category_distribution,
    get_ward_telemetry,
    search_complaints,
    get_complaint_detail,
    execute_olap_operation,
    OLAPRequest,
    get_ml_metrics,
    predict_complaint,
    PredictRequest,
    get_clusters,
    get_association_rules,
    get_recurrence_patterns,
)

results = []

def run_test(test_id, category, feature_name, test_fn):
    t0 = time.perf_counter()
    status = "PASS"
    error_msg = ""
    details = ""
    try:
        details = test_fn()
    except Exception as e:
        status = "FAIL"
        error_msg = str(e)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    results.append({
        "id": test_id,
        "category": category,
        "feature": feature_name,
        "status": status,
        "latency_ms": latency_ms,
        "details": details if status == "PASS" else error_msg
    })

# 1. HTML Page Templates
def test_template_overview():
    from app import templates
    resp = templates.get_template("overview.html").render({"active_page": "overview"})
    assert "Municipal Civic Overview" in resp
    assert "monthlyTrendChart" in resp
    assert "categoryDonutChart" in resp
    assert "filterKW" in resp
    return "Template renders correctly with Chart.js canvases & 1-click ward buttons"

def test_template_explorer():
    from app import templates
    resp = templates.get_template("explorer.html").render({"active_page": "complaint-explorer"})
    assert "Complaint Explorer" in resp
    assert "inspectorContent" in resp
    assert "complaintsTable" in resp
    return "Template renders correctly with master-detail inspector & search"

def test_template_olap():
    from app import templates
    resp = templates.get_template("olap.html").render({"active_page": "olap-laboratory"})
    assert "OLAP Multi-Dimensional Engine" in resp
    assert "1-Click Viva / Lab Demo Showcase" in resp
    assert "Data Warehousing Theory & Concept" in resp
    return "Template renders correctly with 1-click presets & dimensional transform visualizers"

def test_template_ml():
    from app import templates
    resp = templates.get_template("ml_models.html").render({"active_page": "ml-models"})
    assert "Supervised Machine Learning & Grievance Triage" in resp
    assert "loadDemoPreset('pothole')" in resp
    assert "recRiskScore" in resp
    assert "predSlaDays" in resp
    return "Template renders correctly with multi-head predictions & 5 demo presets"

def test_template_patterns():
    from app import templates
    resp = templates.get_template("patterns_and_clusters.html").render({"active_page": "patterns-and-clusters"})
    assert "Civic Patterns, Clusters & Association Rules" in resp
    assert "clusterRadarChart" in resp
    assert "liftSlider" in resp
    assert "wardDossierCard" in resp
    return "Template renders correctly with 5D radar chart, PCA scatter & rule studio"

# 2. Overview APIs
def test_api_kpis():
    kpis = get_overview_kpis()
    assert kpis["total_volume"] == 960000
    assert kpis["active_categories"] == 13
    assert kpis["active_wards"] == 24
    return f"Total: {kpis['total_volume']:,} | Wards: {kpis['active_wards']} | SLA: {kpis['avg_resolution_days']}d"

def test_api_categories():
    data = get_category_distribution()
    assert "categories" in data
    cats = data["categories"]
    assert len(cats) >= 5
    return f"{len(cats)} categories loaded, top: {cats[0]['category_name']} ({cats[0]['percentage']}%)"

def test_api_ward_telemetry():
    data = get_ward_telemetry()
    assert "wards" in data
    wards = data["wards"]
    assert len(wards) == 24
    return f"All {len(wards)} wards verified, highest volume: Ward {wards[0]['ward']} ({wards[0]['total_complaints']:,})"

# 3. Explorer APIs
def test_api_explorer_search_default():
    res = search_complaints(limit=5)
    assert res["total"] > 0
    assert len(res["records"]) <= 5
    first_id = res["records"][0]["complaint_id"]
    return f"Retrieved {len(res['records'])} records (Total: {res['total']:,}), First ID: {first_id}"

def test_api_explorer_filter_ward():
    res = search_complaints(ward="K/W", limit=5)
    assert len(res["records"]) > 0
    for r in res["records"]:
        assert r["ward"] == "K/W"
    return f"Filtered by Ward K/W: verified all {len(res['records'])} records belong to K/W"

def test_api_explorer_text_search():
    res = search_complaints(q="Andheri", limit=5)
    assert len(res["records"]) > 0
    return f"Search 'Andheri' matched {len(res['records'])} records successfully"

def test_api_explorer_detail():
    res = search_complaints(limit=1)
    cid = res["records"][0]["complaint_id"]
    detail = get_complaint_detail(cid)
    assert detail["complaint_id"] == cid
    assert "category_name" in detail
    assert "ward_area" in detail
    return f"Verified deep-dive lookup for Complaint ID {cid} in Ward {detail['ward']}"

# 4. OLAP Operations
def test_olap_rollup():
    req = OLAPRequest(operation="rollup")
    res = execute_olap_operation(req)
    assert res["operation"] == "ROLLUP"
    assert res["row_count"] == 3  # 3 Zones
    assert "granular_wards" in res["extra_meta"]
    assert len(res["extra_meta"]["granular_wards"]) == 24
    return f"Roll-up verified: 24 wards aggregated into 3 zones ({res['row_count']} rows)"

def test_olap_drilldown():
    req = OLAPRequest(operation="drilldown")
    res = execute_olap_operation(req)
    assert res["operation"] == "DRILLDOWN"
    assert res["row_count"] > 0
    assert "annual_total" in res["extra_meta"]
    return f"Drill-down verified: temporal breakdown loaded ({res['row_count']} quarter/month rows)"

def test_olap_slice():
    req = OLAPRequest(operation="slice", slice_category="Drainage Overflow / Flooding")
    res = execute_olap_operation(req)
    assert res["operation"] == "SLICE"
    assert res["row_count"] > 0
    assert res["extra_meta"]["active_category"] == "Drainage Overflow / Flooding"
    return f"Slice verified: 3D plane cut across '{res['extra_meta']['active_category']}' ({res['row_count']} rows)"

def test_olap_dice():
    req = OLAPRequest(operation="dice")
    res = execute_olap_operation(req)
    assert res["operation"] == "DICE"
    assert res["row_count"] > 0
    assert "diced_wards" in res["extra_meta"]
    assert len(res["extra_meta"]["diced_wards"]) == 8
    assert "severity_breakdown" in res["extra_meta"]
    assert len(res["extra_meta"]["severity_breakdown"]) == 3
    return f"Dice sub-cube verified: 3 Cats x 2 Zones x FY2024 ({res['row_count']} rows, 8-Ward drill-down & severity breakdown)"

def test_olap_pivot():
    req = OLAPRequest(operation="pivot")
    res = execute_olap_operation(req)
    assert res["operation"] == "PIVOT"
    assert res["row_count"] > 0
    assert "transposed_matrix" in res["extra_meta"]
    assert len(res["extra_meta"]["transposed_matrix"]) == 4
    return f"Pivot verified: Cross-tabulation heatmap with transposed rotation matrix ({res['row_count']} rows, 4 categories x 5 wards transposed)"

# 5. ML Models & Triage Pipeline
def test_ml_metrics():
    metrics = get_ml_metrics()
    assert "models" in metrics
    assert "LightGBM (Feature Engineered V2)" in metrics["models"]
    lgb_acc = metrics["models"]["LightGBM (Feature Engineered V2)"]["accuracy"]
    return f"Metrics loaded: LightGBM V2 accuracy = {lgb_acc*100:.1f}%, 13 categories evaluated"

def test_ml_preset_pothole():
    req = PredictRequest(
        ward_code="K/W",
        complaint_channel="MyBMC App",
        severity="High",
        is_monsoon_season=1,
        has_photo_evidence=1,
        has_gps_location=1,
        grievance_text="Large deep pothole crater on Link Road near Andheri station"
    )
    res = predict_complaint(req)
    assert res["prediction"] == "Pothole / Road Damage"
    assert res["routing_target"] == "Dept-Roads-K/W"
    assert "recurrence_prediction" in res
    assert "operational_forecast" in res
    return f"Predicted: {res['prediction']} ({res['confidence']}%) | Route: {res['routing_target']} | Recurrence: {res['recurrence_prediction']['repeat_risk_level']}"

def test_ml_preset_drainage():
    req = PredictRequest(
        ward_code="L",
        complaint_channel="1916 Helpline",
        severity="Critical",
        is_monsoon_season=1,
        has_photo_evidence=1,
        has_gps_location=1,
        grievance_text="Severe stormwater drain overflow flooding residential ground floor premises in Kurla"
    )
    res = predict_complaint(req)
    assert res["prediction"] == "Drainage Overflow / Flooding"
    assert res["routing_target"] == "Dept-Storm-L"
    return f"Predicted: {res['prediction']} ({res['confidence']}%) | Route: {res['routing_target']} | SLA Forecast: {res['operational_forecast']['predicted_resolution_days']}d"

def test_ml_preset_garbage():
    req = PredictRequest(
        ward_code="G/N",
        complaint_channel="WhatsApp",
        severity="Medium",
        is_monsoon_season=0,
        has_photo_evidence=1,
        has_gps_location=1,
        grievance_text="Overflowing garbage dump and uncleared debris blocking street corner near metro station in Dharavi"
    )
    res = predict_complaint(req)
    assert res["prediction"] == "Solid Waste / Garbage"
    assert res["routing_target"] == "Dept-Solid-G/N"
    return f"Predicted: {res['prediction']} ({res['confidence']}%) | Route: {res['routing_target']} | SLA Forecast: {res['operational_forecast']['predicted_resolution_days']}d"

def test_ml_preset_water():
    req = PredictRequest(
        ward_code="D",
        complaint_channel="1916 Helpline",
        severity="Critical",
        is_monsoon_season=0,
        has_photo_evidence=1,
        has_gps_location=1,
        grievance_text="High pressure main water supply pipeline rupture leaking thousands of litres across main road near Malabar Hill"
    )
    res = predict_complaint(req)
    assert "Water" in res["prediction"]
    return f"Predicted: {res['prediction']} ({res['confidence']}%) | Route: {res['routing_target']} | Turnaround: {res['operational_forecast']['predicted_resolution_days']}d"

def test_ml_preset_construction():
    req = PredictRequest(
        ward_code="H/W",
        complaint_channel="Online Portal",
        severity="High",
        is_monsoon_season=0,
        has_photo_evidence=1,
        has_gps_location=0,
        grievance_text="Illegal RCC shed construction encroaching public footpath without BMC building sanction near Bandra station"
    )
    res = predict_complaint(req)
    assert "Illegal Construction" in res["prediction"] or "Encroachment" in res["prediction"]
    return f"Predicted: {res['prediction']} ({res['confidence']}%) | Route: {res['routing_target']} | Recurrence: {res['recurrence_prediction']['risk_score_pct']}%"

# 6. Data Mining (Clusters, FP-Growth, Recurrence)
def test_mining_clusters():
    data = get_clusters()
    assert data["k"] == 3
    assert len(data["cluster_profiles"]) == 3
    assert len(data["wards"]) == 24
    return f"k={data['k']} optimal clusters loaded: {len(data['wards'])} wards with 2D PCA spatial coordinates"

def test_mining_association_rules():
    data = get_association_rules()
    assert data["total_rules_found"] > 0
    assert len(data["top_rules"]) > 0
    top_rule = data["top_rules"][0]
    assert "lift" in top_rule
    return f"{data['total_rules_found']} rules loaded | Peak Lift = {top_rule['lift']:.2f}x | Min Support = {data['min_support']}"

def test_mining_recurrence():
    data = get_recurrence_patterns()
    assert "by_category" in data
    assert "prior_distribution" in data
    assert len(data["by_category"]) > 0
    return f"Recurrence dynamics loaded: {len(data['by_category'])} categories & {len(data['prior_distribution'])} prior-case tiers"

# Execution
if __name__ == "__main__":
    tests = [
        ("T01", "UI Templates", "Overview Page Template", test_template_overview),
        ("T02", "UI Templates", "Complaint Explorer Page Template", test_template_explorer),
        ("T03", "UI Templates", "OLAP Laboratory Page Template", test_template_olap),
        ("T04", "UI Templates", "ML Models Page Template", test_template_ml),
        ("T05", "UI Templates", "Patterns & Clusters Page Template", test_template_patterns),
        
        ("T06", "Overview API", "Executive KPIs (/api/overview/kpis)", test_api_kpis),
        ("T07", "Overview API", "Category Distribution (/api/overview/category-distribution)", test_api_categories),
        ("T08", "Overview API", "Ward Telemetry (/api/overview/ward-telemetry)", test_api_ward_telemetry),
        
        ("T09", "Explorer API", "Default Grievance Search (/api/explorer/search)", test_api_explorer_search_default),
        ("T10", "Explorer API", "Ward Filter Search (?ward=K/W)", test_api_explorer_filter_ward),
        ("T11", "Explorer API", "Full-Text Search (?q=Andheri)", test_api_explorer_text_search),
        ("T12", "Explorer API", "Complaint Deep-Dive Detail (/api/explorer/complaint/{id})", test_api_explorer_detail),
        
        ("T13", "OLAP Engine", "Roll-up Operation (Ward -> Zone)", test_olap_rollup),
        ("T14", "OLAP Engine", "Drill-down Operation (Year -> Qtr -> Month)", test_olap_drilldown),
        ("T15", "OLAP Engine", "Slice Operation (3D Plane Cut on Category)", test_olap_slice),
        ("T16", "OLAP Engine", "Dice Operation (Sub-Cube 3x2x1 Filter)", test_olap_dice),
        ("T17", "OLAP Engine", "Pivot Operation (Cross-Tabulation & Transpose)", test_olap_pivot),
        
        ("T18", "ML Pipeline", "Evaluation Metrics (/api/ml/metrics)", test_ml_metrics),
        ("T19", "ML Pipeline", "Live Preset 1: Monsoon Pothole (K/W)", test_ml_preset_pothole),
        ("T20", "ML Pipeline", "Live Preset 2: SWD Drainage Burst (L Kurla)", test_ml_preset_drainage),
        ("T21", "ML Pipeline", "Live Preset 3: Solid Waste Dump (G/N Dharavi)", test_ml_preset_garbage),
        ("T22", "ML Pipeline", "Live Preset 4: Water Pipe Burst (D Malabar)", test_ml_preset_water),
        ("T23", "ML Pipeline", "Live Preset 5: Illegal Demolition (H/W Bandra)", test_ml_preset_construction),
        
        ("T24", "Data Mining", "K-Means Ward Clusters (/api/mining/clusters)", test_mining_clusters),
        ("T25", "Data Mining", "FP-Growth Rules (/api/mining/association-rules)", test_mining_association_rules),
        ("T26", "Data Mining", "Recurrence Patterns (/api/mining/recurrence)", test_mining_recurrence),
    ]

    print("=====================================================================")
    print("           BMC CIVICLENS AUTOMATED END-TO-END VERIFICATION           ")
    print("=====================================================================")

    passed = 0
    failed = 0
    for tid, cat, name, fn in tests:
        run_test(tid, cat, name, fn)
        res = results[-1]
        status_symbol = "[PASS]" if res["status"] == "PASS" else "[FAIL]"
        if res["status"] == "PASS":
            passed += 1
        else:
            failed += 1
        print(f"[{res['id']}] {res['category']:<14} | {res['feature']:<40} | {status_symbol} ({res['latency_ms']:>6.1f}ms)")
        if res["status"] != "PASS":
            print(f"     ERROR: {res['details']}")

    print("=====================================================================")
    print(f"TOTAL TESTS: {len(tests)} | PASSED: {passed} | FAILED: {failed}")
    print("=====================================================================")
    
    with open("outputs/test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved test log to outputs/test_results.json")
