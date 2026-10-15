"""
DWDM Project: Mumbai BMC Ward Clustering Analysis
Multi-Attribute K-Means Clustering across all 24 Administrative Wards:
Features:
- Total complaint density / volume
- Mean resolution days
- Citizen satisfaction rate
- Slum percentage
- Monsoon vulnerability (monsoon/non-monsoon volume ratio)
- Repeat complainant proportion
- High-severity complaint ratio
- Average estimated cost (INR)

Outputs:
- Optimal K selection (Inertia & Silhouette Scores)
- 2D PCA projection for cluster visualization
- Cluster profiles and administrative characterization
- outputs/mining/ward_clusters.json
"""

import json
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pymysql
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

load_dotenv()


def get_db_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "bmc_complaint_dw"),
        port=int(os.getenv("DB_PORT", 3306)),
    )


def extract_ward_metrics():
    conn = get_db_connection()
    query = """
    SELECT 
        l.ward,
        l.ward_area,
        l.zone,
        l.ward_type,
        l.population_density,
        AVG(l.ward_slum_percentage) AS slum_pct,
        COUNT(f.complaint_key) AS total_complaints,
        AVG(f.resolution_days) AS avg_resolution_days,
        AVG(f.citizen_satisfied) AS satisfaction_rate,
        AVG(f.repeat_complainant) AS repeat_complainant_rate,
        AVG(f.estimated_cost_inr) AS avg_estimated_cost,
        SUM(CASE WHEN f.severity IN ('High', 'Critical') THEN 1 ELSE 0 END) / COUNT(f.complaint_key) AS high_severity_ratio,
        SUM(CASE WHEN t.is_monsoon_season = 1 THEN 1 ELSE 0 END) / (COUNT(f.complaint_key) * 0.333) AS monsoon_sensitivity_ratio
    FROM fact_complaint f
    JOIN dim_location l ON f.location_key = l.location_key
    JOIN dim_time t ON f.time_key = t.time_key
    GROUP BY l.ward, l.ward_area, l.zone, l.ward_type, l.population_density
    ORDER BY l.ward;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def run_clustering():
    print("Extracting ward telemetry from Data Warehouse...")
    df = extract_ward_metrics()
    print(f"Loaded {len(df)} administrative wards.")

    feature_cols = [
        "slum_pct",
        "total_complaints",
        "avg_resolution_days",
        "satisfaction_rate",
        "repeat_complainant_rate",
        "avg_estimated_cost",
        "high_severity_ratio",
        "monsoon_sensitivity_ratio",
    ]

    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine optimal K (2 to 5)
    k_metrics = []
    best_k = 3
    best_silhouette = -1.0

    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        inertia = km.inertia_
        k_metrics.append({"k": k, "inertia": round(float(inertia), 2), "silhouette": round(float(sil), 4)})
        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k

    print(f"K-Evaluation: {k_metrics}. Best K by silhouette: {best_k}")

    # Standardize on k=3 or 4 for meaningful municipal segmentation
    final_k = best_k if best_k in [3, 4] else 3
    kmeans = KMeans(n_clusters=final_k, random_state=42, n_init=15)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # Run 2D PCA for spatial map
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)
    df["pca_x"] = [round(float(c[0]), 3) for c in pca_coords]
    df["pca_y"] = [round(float(c[1]), 3) for c in pca_coords]
    variance_explained = [round(float(v), 3) for v in pca.explained_variance_ratio_]

    # Cluster profiles & auto-naming
    cluster_profiles = []
    cluster_names = {
        0: "Tier 1: High-Density High-Stress Corridors",
        1: "Tier 2: Rapid-Resolution Commercial Hubs",
        2: "Tier 3: Monsoon-Sensitive Suburban Enclaves",
        3: "Tier 4: Mixed Infrastructure Baselines",
    }

    for c_id in range(final_k):
        c_sub = df[df["cluster"] == c_id]
        c_name = cluster_names.get(c_id, f"Cluster {c_id}")
        cluster_profiles.append({
            "cluster_id": c_id,
            "label": c_name,
            "ward_count": len(c_sub),
            "wards": c_sub["ward"].tolist(),
            "avg_slum_pct": round(float(c_sub["slum_pct"].mean()), 1),
            "avg_complaints": int(c_sub["total_complaints"].mean()),
            "avg_resolution_days": round(float(c_sub["avg_resolution_days"].mean()), 1),
            "satisfaction_rate": round(float(c_sub["satisfaction_rate"].mean() * 100), 1),
            "repeat_rate": round(float(c_sub["repeat_complainant_rate"].mean() * 100), 1),
            "avg_cost": int(c_sub["avg_estimated_cost"].mean()),
            "monsoon_sensitivity": round(float(c_sub["monsoon_sensitivity_ratio"].mean()), 2),
        })

    # Prepare Ward list for frontend
    wards_data = []
    for _, row in df.iterrows():
        wards_data.append({
            "ward": row["ward"],
            "ward_area": row["ward_area"],
            "zone": row["zone"],
            "ward_type": row["ward_type"],
            "population_density": row["population_density"],
            "slum_pct": round(float(row["slum_pct"]), 1),
            "total_complaints": int(row["total_complaints"]),
            "avg_resolution_days": round(float(row["avg_resolution_days"]), 1),
            "satisfaction_rate": round(float(row["satisfaction_rate"] * 100), 1),
            "repeat_complainant_rate": round(float(row["repeat_complainant_rate"] * 100), 1),
            "avg_cost": int(row["avg_estimated_cost"]),
            "cluster": int(row["cluster"]),
            "pca_x": row["pca_x"],
            "pca_y": row["pca_y"],
        })

    output_payload = {
        "k": final_k,
        "k_evaluation": k_metrics,
        "pca_variance_ratio": variance_explained,
        "cluster_profiles": cluster_profiles,
        "features_used": feature_cols,
        "wards": wards_data,
    }

    os.makedirs(os.path.join("outputs", "mining"), exist_ok=True)
    out_file = os.path.join("outputs", "mining", "ward_clusters.json")
    with open(out_file, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Cluster analysis completed successfully. Output saved to {out_file}")


if __name__ == "__main__":
    run_clustering()
