"""
DWDM Project: Mumbai BMC Civic Complaints Association Rule Mining
Uses FP-Growth algorithm from mlxtend to discover co-occurrence rules:
Attributes:
- Complaint Category
- Severity Level (Low, Medium, High, Critical)
- Channel (App, WhatsApp, Helpline, CFC, etc.)
- Ward Type (South, Central, Suburban)
- Monsoon Seasonality (Monsoon vs Non-Monsoon)
- Resolution Speed (<7 days, 7-20 days, >20 days)
- Citizen Satisfaction (Satisfied vs Unsatisfied)

Outputs:
- outputs/mining/association_rules.json
"""

import json
import os
from dotenv import load_dotenv
import pandas as pd
import pymysql
from mlxtend.frequent_patterns import fpgrowth, association_rules

load_dotenv()


def get_db_connection():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "localhost"),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "bmc_complaint_dw"),
        port=int(os.getenv("DB_PORT", 3306)),
    )


def mine_association_rules(sample_size=100000, min_support=0.03, min_confidence=0.4):
    print(f"Sampling {sample_size} complaints from warehouse for transaction mining...")
    conn = get_db_connection()
    query = f"""
    SELECT 
        c.category_name,
        f.severity,
        f.complaint_channel,
        l.ward_type,
        CASE WHEN t.is_monsoon_season = 1 THEN 'Monsoon Season' ELSE 'Non-Monsoon' END AS season,
        CASE 
            WHEN f.resolution_days <= 7 THEN 'Rapid Resolution (<=7d)'
            WHEN f.resolution_days <= 20 THEN 'Standard Resolution (8-20d)'
            ELSE 'Delayed Resolution (>20d)'
        END AS resolution_bucket,
        CASE WHEN f.citizen_satisfied = 1 THEN 'Citizen Satisfied' ELSE 'Citizen Unsatisfied' END AS satisfaction
    FROM fact_complaint f
    JOIN dim_category c ON f.category_key = c.category_key
    JOIN dim_location l ON f.location_key = l.location_key
    JOIN dim_time t ON f.time_key = t.time_key
    LIMIT {sample_size};
    """
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"Loaded {len(df)} transactions. Transforming into binary incidence matrix...")

    # Build transaction item columns
    transactions = []
    for _, row in df.iterrows():
        txn = [
            f"Cat: {row['category_name']}",
            f"Severity: {row['severity']}",
            f"Channel: {row['complaint_channel']}",
            f"WardType: {row['ward_type']}",
            row["season"],
            row["resolution_bucket"],
            row["satisfaction"],
        ]
        transactions.append(txn)

    # Convert to one-hot encoded DataFrame for mlxtend
    from mlxtend.preprocessing import TransactionEncoder

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

    print(f"Running FP-Growth with min_support={min_support}...")
    frequent_itemsets = fpgrowth(df_onehot, min_support=min_support, use_colnames=True)
    print(f"Discovered {len(frequent_itemsets)} frequent itemsets.")

    print(f"Generating association rules with min_confidence={min_confidence}...")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    print(f"Extracted {len(rules)} association rules. Sorting by lift...")

    # Filter rules to those with lift > 1.05 and at least 1 antecedent and 1 consequent
    rules = rules[rules["lift"] > 1.05].sort_values(by="lift", ascending=False)

    rules_list = []
    for _, row in rules.head(50).iterrows():
        antecedents = list(row["antecedents"])
        consequents = list(row["consequents"])
        rules_list.append({
            "antecedents": antecedents,
            "consequents": consequents,
            "support": round(float(row["support"]), 4),
            "confidence": round(float(row["confidence"]), 4),
            "lift": round(float(row["lift"]), 4),
            "conviction": round(float(row["conviction"]), 4) if row["conviction"] != float("inf") else 999.0,
            "rule_str": f"{', '.join(antecedents)} => {', '.join(consequents)}",
        })

    os.makedirs(os.path.join("outputs", "mining"), exist_ok=True)
    out_file = os.path.join("outputs", "mining", "association_rules.json")
    with open(out_file, "w") as f:
        json.dump({
            "total_transactions_mined": len(df),
            "min_support": min_support,
            "min_confidence": min_confidence,
            "total_rules_found": len(rules),
            "top_rules": rules_list,
        }, f, indent=2)

    print(f"Saved {len(rules_list)} high-lift association rules to {out_file}")


if __name__ == "__main__":
    mine_association_rules()
