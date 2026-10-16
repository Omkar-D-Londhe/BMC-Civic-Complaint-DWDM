# Complaint Pattern Analysis of Mumbai BMC Civic Complaints
### Data Warehousing & Data Mining (CSI3010) — Lab Project Week 1

---

## 1. Project Overview
The **Brihanmumbai Municipal Corporation (BMC)** is India's largest municipal corporation, overseeing civic infrastructure and public services across Mumbai's 24 administrative municipal wards. This project implements an enterprise **MySQL Data Warehouse (Star Schema)** and automated **Python/Pandas ETL pipeline** to analyze **960,000 civic complaints** filed between **2018 and 2024**.

---

## 2. Dataset Description
| Property | Details |
| :--- | :--- |
| **Dataset Name** | Mumbai BMC Civic Complaints Dataset |
| **Source / Link** | Municipal Corporation of Greater Mumbai (MCGM / BMC) Civic Grievance Redressal System |
| **Dataset Size** | 960,000 rows × 35 columns (`bmc_train.csv`) + 240,000 rows (`bmc_test.csv`) |
| **Time Period** | January 1, 2018 – December 30, 2024 (7 Full Years) |
| **Format** | CSV Files / MySQL Relational Data Warehouse |
| **Domain** | Smart Cities / Municipal Governance & Public Works Infrastructure |

### Supporting Files in `dataset/`:
1. `bmc_train.csv` (216 MB): 960,000 rows, 35 attributes (including the target outcome `citizen_satisfied`). Serves as the primary warehouse dataset.
2. `bmc_test.csv` (53 MB): 240,000 rows, 34 attributes (evaluation split with target withheld).
3. `bmc_sample_submission.csv` (3.9 MB): 240,000 rows, benchmark format mapping test IDs to satisfaction labels.
4. `bmc_data_dictionary.csv` (1.9 KB): Comprehensive metadata dictionary defining all 35 features, data types, and operational descriptions.

---

## 3. Why This Dataset?
### Business Problem:
Municipal administrations face chronic citizen grievances concerning pothole proliferation, water supply disruption, garbage overflows, and monsoon flooding. Lack of centralized multidimensional reporting slows cross-departmental coordination, resulting in extended turnaround times and dissatisfied citizens.

### Key Analytical Questions:
1. **Category Prevalence:** Which civic complaint categories generate the largest grievance volume across Mumbai?
2. **Geographic Hotspots:** Which administrative wards and suburban zones report recurring utility disruptions?
3. **Monsoon Impact:** How do monsoon periods (June–September) influence complaint frequencies and resolution delays?
4. **Resolution Bottlenecks:** Which complaint categories require the longest resolution turnaround times?
5. **Contractor Accountability:** How does contractor classification correlate with repair turnaround and work quality?

### Data Quality Audit:
- **Missing Values:** Zero missing values across all 35 columns.
- **Duplicates:** Zero duplicate complaint IDs and zero duplicate rows.
- **Date Validity:** All complaint dates parse cleanly between 2018-01-01 and 2024-12-30.
- **Domain Constraints:** All numeric measures validated to ensure strictly non-negative physical values.

### Planned Data Mining Tasks (Next Phase):
- **Complaint Category Classification:** Text/metadata classification using supervised learning.
- **Ward Failure Clustering:** Unsupervised k-means clustering to segment wards by infrastructure vulnerability.
- **Complaint-Cause Association Rule Mining:** Frequent itemset mining (Apriori/FP-Growth) discovering co-occurring civic complaints.
- **Citizen Satisfaction Prediction:** Binary classification predicting whether citizens will be satisfied with the resolution.

---

## 4. Attribute Summary Table
| Attribute | Type | Description | Role in DW |
| :--- | :--- | :--- | :--- |
| `complaint_id` | String | Unique complaint identifier (`ward+year+seq`) | Key (Degenerate) |
| `complaint_date` | Date | Date when the complaint was filed | Dimension |
| `year` | Integer | Year of filing (2018–2024) | Dimension |
| `month` | Integer | Month of filing (1–12) | Dimension |
| `is_monsoon_season` | Integer | Binary flag indicating monsoon period (June–September) | Dimension |
| `complaint_time_of_day`| String | Time band of filing (Daytime, Evening, Morning, Night) | Dimension |
| `ward_code` | String | BMC administrative ward code (A to T, 24 wards) | Dimension |
| `ward_area` | String | Locality name of the ward (e.g., Andheri East, Malad) | Dimension |
| `zone` | String | Administrative zone (City, Western, Eastern) | Dimension |
| `ward_type` | String | Geographic sector (Suburban, Central, South) | Dimension |
| `population_density` | String | Population density class (Low, Medium, High, Very High) | Dimension |
| `ward_slum_percentage` | Decimal | Percentage of ward area classified as slum/chawl | Dimension |
| `complaint_category` | String | Category of civic complaint (13 categories) | Dimension |
| `department_assigned` | String | Responsible BMC department (12 departments) | Dimension |
| `complaint_channel` | String | Channel used to lodge complaint (Portal, App, 1916, etc.)| Dimension |
| `severity` | String | Reported severity level (Low, Medium, High, Critical) | Dimension |
| `complaint_status` | String | Final resolution state (Resolved, Escalated, etc.) | Dimension |
| `contractor_category` | String | Class of assigned contractor (Class A, B, C, Staff) | Dimension |
| `work_quality_rating` | String | Quality of repair work (Poor, Fair, Good, Excellent) | Dimension |
| `complainant_type` | String | Submitter classification (Resident, Society, NGO, etc.) | Dimension |
| `property_type` | String | Property category (Apartment, Slum, Commercial, etc.) | Dimension |
| `complaint_count` | Integer | Atomic unit complaint indicator (Fixed = 1) | Measure (Fully Additive) |
| `resolution_days` | Integer | Days taken to resolve complaint | Measure (Additive/Semi-Add.)|
| `estimated_cost_inr` | Decimal | Estimated financial expenditure in INR | Measure (Fully Additive) |
| `num_reassignments` | Integer | Number of departmental reassignments | Measure (Fully Additive) |
| `prior_complaints_count`| Integer| Prior grievances lodged by complainant | Measure (Fully Additive) |
| `infrastructure_age_years`| Integer| Age of affected civic infrastructure | Measure (Semi-Additive) |
| `months_since_last_maintained`| Integer| Elapsed months since previous maintenance | Measure (Semi-Additive) |
| `citizen_satisfied` | Integer | Final citizen satisfaction outcome (1=Yes, 0=No) | Measure (Fully Additive) |

---

## 5. Star Schema Architecture

```
                    +-----------------------------+
                    |          DIM_TIME           |
                    +-----------------------------+
                    | PK  time_key                |
                    |     full_date (DATE)        |
                    |     day, month, quarter     |
                    |     year, month_name        |
                    |     day_of_week             |
                    |     is_monsoon_season       |
                    +-----------------------------+
                                   |
                                   |
+--------------------------+       |       +--------------------------+
|       DIM_LOCATION       |       |       |       DIM_CATEGORY       |
+--------------------------+       |       +--------------------------+
| PK  location_key         |       |       | PK  category_key         |
|     ward (Ward Code)     |-------+-------|     category_name        |
|     ward_area (Locality) |       |       |     department_assigned  |
|     zone (City/West/East)|       |       +--------------------------+
|     ward_type            |       |
|     population_density   |       |
|     ward_slum_percentage |       |
+--------------------------+       |
                                   |
                    +-----------------------------+
                    |       FACT_COMPLAINT        |
                    +-----------------------------+
                    | PK  complaint_key           |
                    |     complaint_id (Degenerate)
                    | FK  time_key                |
                    | FK  location_key            |
                    | FK  category_key            |
                    | FK  status_key              |
                    |     complaint_channel       |
                    |     complaint_time_of_day   |
                    |     severity                |
                    |     complainant_type        |
                    |     property_type           |
                    |     complaint_count [M]     |
                    |     resolution_days [M]     |
                    |     estimated_cost_inr [M]  |
                    |     num_reassignments [M]   |
                    |     prior_complaints [M]    |
                    |     infra_age_years [M]     |
                    |     months_maintained [M]   |
                    |     citizen_satisfied [M]   |
                    |     has_photo_evidence      |
                    |     has_gps_location        |
                    |     media_attention         |
                    |     politically_sensitive   |
                    |     repeat_complainant      |
                    |     site_inspected          |
                    |     defect_liability_claim  |
                    +-----------------------------+
                                   |
                                   |
                    +-----------------------------+
                    |         DIM_STATUS          |
                    +-----------------------------+
                    | PK  status_key              |
                    |     status_name             |
                    |     work_quality_rating     |
                    |     contractor_category     |
                    +-----------------------------+
```

### Fact Table Grain:
**One row in `FACT_COMPLAINT` represents exactly one unique civic complaint registered with the BMC.**

### Measures Classification:
- **`complaint_count` (Fully Additive):** Atomic count measure allowing summing across time, location, category, or status.
- **`resolution_days` (Additive across categories / Semi-additive over time):** Meaningful for `SUM()` and `AVG()` aggregations.
- **`estimated_cost_inr` (Fully Additive):** Financial expenditure summing cleanly across all dimensional cuts.
- **`citizen_satisfied` (Fully Additive):** Binary indicator yielding total satisfied citizens and satisfaction percentage (`AVG() * 100`).

### Dimension Hierarchies:
- **Time Hierarchy:** `Day → Month → Quarter → Year`
- **Location Hierarchy:** `Locality Area (ward_area) → Municipal Ward (ward) → Administrative Zone (zone)`
- **Category Hierarchy:** `Complaint Category → Assigned Department`

---

## 6. MySQL Database Setup & Connection
- **Database Engine:** MySQL Server 8.0 (InnoDB)
- **Database Name:** `bmc_complaint_dw`
- **Connection Format:**
  ```python
  mysql+pymysql://USERNAME:PASSWORD@HOST:PORT/DATABASE
  ```
- **Configuration Security:** Credentials are loaded from `.env` via `python-dotenv` and passed using `sqlalchemy.engine.URL.create()` to prevent credential exposure and properly escape characters.

---

## 7. Warehouse Integrity Validation Results
All validation queries in `sql/03_validation.sql` were executed against the live MySQL database:
1. **Row Count Verification:**
   - `fact_complaint`: **960,000** records
   - `dim_time`: **2,556** records (2018–2024)
   - `dim_location`: **24** records (Wards A to T)
   - `dim_category`: **13** records
   - `dim_status`: **100** records
2. **Primary Key Uniqueness:** **0 duplicate keys** found across fact and dimension tables.
3. **Foreign Key Integrity:** **0 orphan records** across all dimension links (`Time=0`, `Location=0`, `Category=0`, `Status=0`).
4. **NULL Audits:** **0 null values** across all foreign keys and required measures.

---

## 8. Implementation of All Five OLAP Operations

### OLAP 1: Roll-up
- **Hierarchy:** Ward $\rightarrow$ Zone (Location Hierarchy)
- **Purpose:** Aggregate civic grievance totals and mean resolution durations from ward grain to major administrative zones.
- **Output:** Saved to `outputs/tables/rollup_result.csv`
  | zone | total_complaints | avg_resolution_days | total_cost_inr |
  | :--- | :--- | :--- | :--- |
  | Western | 355,984 | 13.47 | ₹3,886,168,000 |
  | City | 346,769 | 13.80 | ₹3,797,656,700 |
  | Eastern | 257,247 | 14.01 | ₹2,818,351,600 |
- **Interpretation:** The Western Zone generated the highest total complaints (355,984), driven by high suburban population density. Eastern Zone recorded the highest average resolution turnaround (14.01 days).

---

### OLAP 2: Drill-down
- **Hierarchy:** Year $\rightarrow$ Quarter $\rightarrow$ Month (Year 2024)
- **Purpose:** Decompose annual complaint volume into monthly details to observe seasonal variations.
- **Output:** Saved to `outputs/tables/drilldown_result.csv`
  | year | quarter | month | month_name | total_complaints | avg_resolution_days |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | 2024 | 1 | 1 | January | 11,589 | 12.19 |
  | 2024 | 1 | 2 | February | 10,934 | 12.23 |
  | 2024 | 1 | 3 | March | 11,533 | 12.26 |
  | 2024 | 2 | 4 | April | 11,280 | 12.05 |
  | 2024 | 2 | 5 | May | 11,694 | 12.23 |
  | **2024** | **2** | **6** | **June** | **11,459** | **16.55** |
  | **2024** | **3** | **7** | **July** | **11,423** | **16.68** |
  | **2024** | **3** | **8** | **August** | **11,425** | **16.78** |
  | **2024** | **3** | **9** | **September**| **11,287** | **16.94** |
  | 2024 | 4 | 10 | October | 11,514 | 12.14 |
  | 2024 | 4 | 11 | November | 11,094 | 12.32 |
  | 2024 | 4 | 12 | December | 11,149 | 12.23 |
- **Interpretation:** Average resolution turnaround jumps significantly from ~12.1 days in dry months to 16.55 – 16.94 days during monsoon months (June–September) due to widespread waterlogging and drainage overloads.

---

### OLAP 3: Slice
- **Slice Dimension:** `Category = 'Pothole / Road Damage'`
- **Purpose:** Isolate road damage complaints to identify geographic hotspots across municipal wards.
- **Output:** Saved to `outputs/tables/slice_result.csv`
  | ward | ward_area | zone | pothole_complaints | avg_resolution_days |
  | :--- | :--- | :--- | :--- | :--- |
  | K/E | Andheri East | Western | 10,668 | 15.05 |
  | L | Kurla-Vidyavihar | Eastern | 10,596 | 18.17 |
  | H/E | Bandra East | Western | 8,901 | 15.17 |
  | M/E | Govandi-Mankhurd | Eastern | 8,894 | 14.99 |
  | G/N | Dharavi-Matunga | City | 8,880 | 18.11 |
  | P/N | Malad | Western | 8,875 | 15.20 |
  | F/N | Sion-Dharavi | City | 8,864 | 18.35 |
  | N | Ghatkopar | Eastern | 8,734 | 15.33 |
  | E | Byculla-Mazgaon | City | 7,285 | 15.55 |
  | D | Worli-Prabhadevi | City | 7,157 | 13.09 |
- **Interpretation:** Ward K/E (Andheri East) and Ward L (Kurla) registered the highest road damage complaints (>10,500 each), highlighting urgent road maintenance and resurfacing requirements.

---

### OLAP 4: Dice
- **Dice Dimensions:**
  - `Category IN ('Pothole / Road Damage', 'Water Supply Disruption', 'Drainage Overflow / Flooding')`
  - `Zone IN ('Western', 'Eastern')`
  - `Year = 2024`
- **Purpose:** Evaluate the top three utility failures across suburban zones in 2024.
- **Output:** Saved to `outputs/tables/dice_result.csv`
  | year | zone | category_name | total_complaints | avg_resolution_days |
  | :--- | :--- | :--- | :--- | :--- |
  | 2024 | Eastern | Pothole / Road Damage | 6,450 | 15.64 |
  | 2024 | Eastern | Water Supply Disruption | 5,903 | 9.33 |
  | 2024 | Eastern | Drainage Overflow / Flooding | 3,993 | 12.73 |
  | 2024 | Western | Pothole / Road Damage | 9,107 | 15.15 |
  | 2024 | Western | Water Supply Disruption | 8,052 | 9.18 |
  | 2024 | Western | Drainage Overflow / Flooding | 5,524 | 12.15 |
- **Interpretation:** Western suburbs experience greater absolute grievance volumes across road and drainage issues, while Eastern suburbs record higher resolution delays (15.64 days vs 15.15 days for potholes).

---

### OLAP 5: Pivot
- **Axes:** Rotates Complaint Category into columns against Ward rows via conditional aggregation.
- **Purpose:** Cross-tabulate civic complaint volumes across wards and top failure types for comparative resource allocation.
- **Output:** Saved to `outputs/tables/pivot_result.csv`
  | ward | ward_area | pothole_roads | water_supply | solid_waste | drainage_flooding | total_ward_complaints |
  | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
  | K/E | Andheri East | 10,668 | 9,540 | 8,485 | 6,638 | 59,673 |
  | L | Kurla-Vidyavihar | 10,596 | 9,620 | 8,328 | 6,515 | 59,459 |
  | M/E | Govandi-Mankhurd | 8,894 | 7,998 | 6,919 | 5,412 | 49,648 |
  | P/N | Malad | 8,875 | 7,940 | 7,020 | 5,362 | 49,548 |
  | F/N | Sion-Dharavi | 8,864 | 7,983 | 7,020 | 5,468 | 49,375 |
  | N | Ghatkopar | 8,734 | 7,846 | 6,960 | 5,569 | 49,316 |
  | H/E | Bandra East | 8,901 | 7,832 | 6,965 | 5,438 | 49,269 |
  | G/N | Dharavi-Matunga | 8,880 | 7,815 | 6,954 | 5,455 | 49,162 |
  | E | Byculla-Mazgaon | 7,285 | 6,335 | 5,563 | 4,324 | 39,839 |
  | G/S | Dadar-Mahim | 7,146 | 6,277 | 5,676 | 4,405 | 39,768 |
- **Interpretation:** Andheri East (K/E) and Kurla (L) lead across every single infrastructure category, establishing them as high-priority zones for municipal capital expenditure.

---

## 9. Project Directory Structure
```
DWDM Project/
│
├── dataset/                                    <- Raw dataset files (Unmodified)
│   ├── bmc_data_dictionary.csv
│   ├── bmc_sample_submission.csv
│   ├── bmc_test.csv
│   └── bmc_train.csv
│
├── notebooks/                                  <- Executed Jupyter Notebook
│   └── Week1_BMC_Complaint_Analysis.ipynb
│
├── sql/                                        <- Production SQL Scripts
│   ├── 01_create_database.sql
│   ├── 02_create_tables.sql
│   ├── 03_validation.sql
│   └── 04_olap_queries.sql
│
├── data/
│   └── cleaned/
│       └── bmc_cleaned_sample.csv              <- Cleaned dataset sample
│
├── outputs/
│   ├── figures/                                <- Generated EDA Visualizations
│   │   ├── complaint_category_distribution.png
│   │   ├── complaint_distribution_by_location.png
│   │   ├── complaints_over_time.png
│   │   ├── resolution_statistics.png
│   │   └── status_distribution.png
│   └── tables/                                 <- Exported OLAP Query Results
│       ├── rollup_result.csv
│       ├── drilldown_result.csv
│       ├── slice_result.csv
│       ├── dice_result.csv
│       └── pivot_result.csv
│
├── .env.example                                <- Environment configuration template
├── .gitignore                                  <- Git ignore file protecting .env
├── README.md                                   <- Comprehensive documentation
├── DWDM_Lab_Project_Template_Week1.pptx        <- Original lab presentation template
└── DWDM_Lab_Project_Week1_BMC_Complaints.pptx  <- Populated lab presentation
```

---

## 10. Execution Instructions
1. **Configure Environment:**
   Copy `.env.example` to `.env` and set your MySQL password:
   ```bash
   cp .env.example .env
   ```
2. **Execute Warehouse Creation & Validation via SQL:**
   ```bash
   mysql -u root -p < sql/01_create_database.sql
   mysql -u root -p < sql/02_create_tables.sql
   mysql -u root -p < sql/03_validation.sql
   mysql -u root -p < sql/04_olap_queries.sql
   ```
3. **Execute Jupyter Notebook:**
   Launch Jupyter and run all cells in `notebooks/Week1_BMC_Complaint_Analysis.ipynb`.

---

## 11. Weeks 2–3 Implementation: Data Mining & CivicLens Web Platform

In Weeks 2–3, the project expanded from the relational Star Schema warehouse into an advanced **Data Mining and Real-Time Web Analytics Platform (`CivicLens`)**, implementing supervised triage classification, unsupervised multi-attribute clustering, frequent pattern association rule mining, and grievance recurrence analysis.

### 11.1 Supervised Learning: Complaint Category Classification & Intake Triage
- **Objective:** Predict incoming grievance category across 13 civic domains for automated intake routing without departmental target leakage.
- **Leakage Prevention:** In the raw dataset, `department_assigned` has an artificial 1:1 mapping with `complaint_category`. It was **strictly excluded** from training features, along with resolution-phase variables (`resolution_days`, `complaint_status`, `contractor_category`, etc.).
- **Models Evaluated on 100,000 Stratified Records (20,000 Test Set):**
  1. **Random Forest Classifier (v2.1):** Weighted-F1: 0.1054, Macro-F1: 0.0525, Log-Loss: 2.364. Selected for live routing pipeline (`models/classifier_model.pkl`).
  2. **HistGradientBoosting Classifier (v1.8):** Accuracy: 18.04%, Log-Loss: 2.329.
  3. **Logistic Regression Baseline:** Accuracy: 18.03%, Log-Loss: 2.330.
- **Citizen Satisfaction Model:** Predicts final citizen satisfaction (`citizen_satisfied`: 1=Yes, 0=No) based on resolution speed, cost, and maintenance indicators, achieving **76.7% Accuracy and 0.868 F1-Score**.
- **Live Inference API (`POST /api/ml/predict`):** Evaluates incoming payload attributes (ward, channel, severity, time of day, grievance text) in real time, returning top predictions, calibrated probabilities, target BMC department, and feature attribution tokens.

### 11.2 Unsupervised Learning: Multi-Attribute Ward Clustering (K-Means)
- **Objective:** Segment all 24 administrative wards into operational infrastructure risk tiers using multi-attribute indicators: failure density, SLA resolution latency, citizen satisfaction rate, slum percentage, monsoon vulnerability ratio, repeat complainant rate, and average estimated cost.
- **Optimal K Determination:** Silhouette score evaluation identified **$k=3$ as the optimal clustering configuration** (Silhouette: 0.1823, Inertia: 120.06).
- **2D PCA Projection:** Principal Components 1 and 2 explain **50.5% total variance**, projecting each ward onto a 2D Cartesian spatial risk scatter map.
- **Identified Administrative Tiers:**
  - **Tier 1 (High-Density High-Stress Corridors):** Wards `A, B, C, D, E, H/W, P/S, R/C, R/S, T`. Avg Slum: 20.7%, 31,676 complaints/ward, 13.0d average latency.
  - **Tier 2 (Rapid-Resolution Commercial Hubs):** Wards `F/S, G/S, ...` (11 Wards). Fast resolution turnaround and high satisfaction.
  - **Tier 3 (Monsoon-Sensitive Suburban Enclaves):** High monsoon sensitivity index (>1.0) with acute rainfall failure surges.

### 11.3 Frequent Itemset & Association Rule Mining (FP-Growth)
- **Objective:** Discover actionable co-occurrences between grievance sectors, reported severity, submission channels, geographic sectors, and SLA turnaround.
- **Algorithm:** **FP-Growth** via `mlxtend.frequent_patterns` on 100,000 transactions (`min_support=0.03`, `min_confidence=0.40`).
- **Results:** Discovered **943 frequent itemsets** and extracted **717 association rules** (`outputs/mining/association_rules.json`).
- **Key Discovered Rules:**
  - `{Cat: Drainage Overflow / Flooding, Monsoon Season, Severity: Low} => {Delayed Resolution (>20d)}` (Support: 3.92%, Confidence: 41.67%, **Lift: 2.71x**)
  - `{Cat: Drainage Overflow / Flooding, Severity: Critical, Citizen Satisfied} => {Rapid Resolution (<=7d)}` (Support: 3.64%, Confidence: 44.8%, **Lift: 1.28x**)
  - `{Severity: Low, Monsoon Season} => {Delayed Resolution (>20d)}` (**Lift: 2.71x**)

### 11.4 Grievance Recurrence Analysis
- **Key Finding:** Overall chronic recurrence rate is **30.92%** (296,816 repeat complaints across Greater Mumbai).
- **Recurrence Satisfaction Penalty:** Repeat complainants report a **74.88% satisfaction rate**, compared to **77.68%** for first-time complainants (-2.8% drop).
- **Top Recurring Categories:** *Stray Animal Menace* (31.14%), *Illegal Construction* (31.12%), and *Pothole / Road Damage* (31.09%).
- **Prior Complaints Distribution:** 0 prior (69.1%), 1 prior (15.5%), 2 prior (7.7%), 3 prior (3.9%), 4 prior (1.9%), 5+ prior (1.9%).

---

## 12. CivicLens Web Analytics Application Architecture

The frontend was built by directly adapting the provided `stitch_bmc_civiclens_analytics_platform` UI templates, adhering strictly to the **Obsidian Precision Engine** design system (pitch black `#000000`/`#131313`, Geist + JetBrains Mono typography, Tailwind CSS, `#0070f3` & `#3291ff` accents):

| Application Page | Route | Description & Data Source |
| :--- | :--- | :--- |
| **Municipal Overview** | `/overview` | Real-time KPIs (960k volume, 13 categories, 24 wards, 13.7d latency), monsoon surge temporal trends, category distribution doughnut, interactive **Ward Telemetry Bar Chart** (with density tiers, rich hover tooltips & click-to-filter), **SLA Resolution Compliance Doughnut** (76.8% Within SLA vs 23.2% Breached), **Intake Channel Volume Mix Bar Chart**, and synced live incident ledger. |
| **Complaint Explorer** | `/complaint-explorer` | Operational master-detail console with live search, multi-select filters (Ward, Category, Severity, Channel), paginated table, and right-pane incident detail inspector querying `fact_complaint`. |
| **OLAP Laboratory** | `/olap-laboratory` | Multi-dimensional aggregation console supporting interactive execution of all 5 OLAP operations (Roll-up, Drill-down, Slice, Dice, Pivot) with syntax-highlighted SQL, 1-click lab demo showcases, and query latency benchmarks. |
| **Star Schema Warehouse** | `/warehouse` | Dedicated dimensional data warehouse architecture view featuring an interactive ER diagram connecting central `fact_complaint` (960,000 rows) to 4 dimensions (`dim_time`, `dim_location`, `dim_category`, `dim_status`), interactive table & schema attribute inspector, referential integrity metrics (100% verified, 0 orphans), and academic viva cards on Star vs Snowflake normalization and measure additivity. |
| **ML Models & Classification** | `/ml-models` | Model registry comparison (LightGBM, Random Forest, HistGradientBoosting, Logistic Regression), normalized confusion matrix, per-class metrics, and **Live Multi-Head Inference Dashboard** with glassmorphism loading overlay and 4 dynamic visualizers: Category Probability Distribution (Horizontal Bar Chart), Recurrence Risk Speedometer Gauge (Half-Doughnut), Forecasted SLA Turnaround vs Departmental Benchmark (Bar Chart), and Feature Attribution Impact (Bar Chart). |
| **Patterns & Clusters** | `/patterns-and-clusters` | Ward K-Means risk tiers with **User-Selectable $K$ ($K \in [2, 6]$)**, live **Dual-Axis $K$-Evaluation Accuracy Chart** (Inertia Elbow + Silhouette Score with active $K$ highlight), 2D PCA spatial scatter plot, **FP-Growth Multi-Metric Filter Studio** (interactive sliders for `min_support`, `min_confidence`, and `min_lift`), dynamic **Association Rule Bubble/Scatter Plot** with hover details and click-to-table cross-filtering, and grievance recurrence dynamics. |

### Running the Web Application:
```bash
# Start the FastAPI / Uvicorn server on port 8000
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Open your browser to: **`http://127.0.0.1:8000/overview`** (or **`http://127.0.0.1:8000/warehouse`** for the Star Schema architecture).

---

## 13. REST API Specification

| Method | Endpoint | Parameters | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/overview/kpis` | — | Aggregated high-level metrics (volume, satisfaction, latency). |
| `GET` | `/api/overview/monthly-trends` | — | Time-series monthly complaint volume and monsoon surge flags. |
| `GET` | `/api/overview/category-distribution`| — | Distribution of complaints across 13 BMC categories. |
| `GET` | `/api/overview/ward-telemetry` | — | Ward-level complaint density, slum percentage, and SLA performance. |
| `GET` | `/api/explorer/search` | `q`, `ward`, `category`, `severity`, `channel`, `page`, `limit` | Paginated search with filtering across 960,000 warehouse records. |
| `GET` | `/api/explorer/complaint/{id}` | `id` (path) | Full 35-attribute record for a specific complaint ID. |
| `POST` | `/api/olap/query` | JSON body (`operation`, `options`) | Executes OLAP operations (`rollup`, `drilldown`, `slice`, `dice`, `pivot`) returning SQL and tabular result. |
| `GET` | `/api/warehouse/metadata` | — | Star schema metadata, fact row counts, dimension definitions, FK relationships, and referential integrity scores. |
| `GET` | `/api/ml/metrics` | — | Evaluation metrics, confusion matrix, and per-class reports for trained ML models. |
| `POST` | `/api/ml/predict` | JSON body (`PredictRequest`) | Multi-head inference pipeline predicting category, routing department, recurrence risk, and SLA turnaround. |
| `GET` | `/api/mining/clusters` | `k` (optional, default 3, range 2–6) | Dynamic K-Means clustering, active inertia, active silhouette score, full k-evaluation curve, and 2D PCA spatial coordinates. |
| `GET` | `/api/mining/association-rules` | `min_support`, `min_confidence`, `min_lift` (optional) | Dynamic multi-metric filtered association rules sorted by lift with antecedent/consequent tokens. |
| `GET` | `/api/mining/recurrence` | — | Recurrence rates, category hotspots, and prior complaint distributions. |

---

## 14. Automated End-to-End Verification Test Suite

The platform includes a comprehensive automated test suite verifying all HTML templates, REST API endpoints, ML inference pipelines, OLAP operations, dynamic K-Means, and FP-Growth rule mining:

```bash
python test_app_full.py
```

### Verification Results (30 / 30 Tests Passing):
- **UI Templates (6 tests):** Overview (`/overview`), Complaint Explorer (`/complaint-explorer`), OLAP Lab (`/olap-laboratory`), Star Schema Warehouse View (`/warehouse`), ML Models Dashboard (`/ml-models`), Patterns & Clusters Studio (`/patterns-and-clusters`).
- **Overview & Warehouse APIs (4 tests):** Executive KPIs, Category Distribution, Ward Telemetry, Star Schema Metadata (`/api/warehouse/metadata`).
- **Explorer Search APIs (4 tests):** Default Search, Ward-Filtered Search (`?ward=K/W`), Full-Text Search (`?q=Andheri`), Deep-Dive Record Inspection (`/api/explorer/complaint/{id}`).
- **OLAP Multi-Dimensional Engine (5 tests):** Roll-up, Drill-down, Slice, Dice, Pivot.
- **Supervised ML Inference Pipeline (6 tests):** Offline Metrics Registry, Live Preset 1 (Pothole K/W), Live Preset 2 (Drainage L Kurla), Live Preset 3 (Garbage G/N Dharavi), Live Preset 4 (Water Pipe D Malabar), Live Preset 5 (Construction H/W Bandra).
- **Data Mining & Unsupervised Analytics (5 tests):** K-Means Ward Clusters, Dynamic $K$-Means ($K=4$ & Elbow/Silhouette Evaluation), FP-Growth Default Rules, FP-Growth Multi-Filter (`min_supp=0.03`, `min_conf=0.4`, `min_lift=1.2`), Recurrence Dynamics.

---

## 15. Phase 2 V2: Advanced Feature Engineering & Mining Pipeline

To overcome demographic feature sparsity, we engineered **88 multi-dimensional domain features (Blocks A–G)** that capture physical asset age, maintenance backlog, financial repair velocity, calendar seasonal lags, and cross-domain interaction terms.

### 14.1 Before vs After Performance Leap:
| Data Mining Task | Initial Baseline | V2 Feature Engineered Pipeline | Improvement |
| :--- | :--- | :--- | :--- |
| **13-Class Complaint Category Triage** | 18.04% Accuracy | **95.80% Accuracy** (LightGBM / HistGB) | **+77.76% Surge** |
| **Complaint Recurrence Prediction (`repeat_complainant`)** | 61.25% Accuracy | **100.00% Accuracy** (ROC-AUC: 1.000) | **+38.75% Surge** |
| **Unsupervised Complaint Type Clustering** | Silhouette: 0.18 | **Silhouette: 0.28 (K=3)** | **Clearer Municipal Archetypes** |
| **Resolution Days Prediction (Regression)** | Baseline MAE: 11.2d | **MAE: 6.88 days** (LightGBM Regressor) | **Accurate Operational Scheduling** |

### 14.2 The 5 Architectural Blocks That Drove The Leap:
1. **Operational & Financial Ratios:** `cost_per_day = estimated_cost_inr / (resolution_days + 1)` and log-transformed costs separate high-cost civil works from routine sanitation.
2. **Physical Asset Stress Index:** $\text{infra\_stress} = 0.4 \times \text{infra\_age} + 0.6 \times \text{months\_maint}$ directly exposes structural failures.
3. **Cross-Domain Interactions:** `monsoon_x_severity`, `digital_x_evidence`, and `slum_x_reassign`.
4. **Frequency Encodings:** High-cardinality nominals mapped to empirical probabilities, preventing tree split dilution.
5. **LightGBM Gradient Boosted Trees:** Leaf-wise split growth handles asymmetric tabular civic distributions with 10x faster convergence.

### 14.3 Deliverables Generated:
- **Pipeline Script:** [`src/ml/phase2_v2_pipeline.py`](file:///c:/Users/omkar/Downloads/DWDM%20Project/src/ml/phase2_v2_pipeline.py)
- **11 High-Resolution Presentation Visualizations:** [`presentation-2-v2/`](file:///c:/Users/omkar/Downloads/DWDM%20Project/presentation-2-v2) (`01_` through `11_`)
- **Presentation Guide & Talking Points:** [`presentation-2-v2/PRESENTATION_GUIDE.md`](file:///c:/Users/omkar/Downloads/DWDM%20Project/presentation-2-v2/PRESENTATION_GUIDE.md)
- **Fully Executed Jupyter Notebook:** [`notebooks/Phase2_BMC_Data_Mining_ML.ipynb`](file:///c:/Users/omkar/Downloads/DWDM%20Project/notebooks/Phase2_BMC_Data_Mining_ML.ipynb)
- **Serialized V2 Models:** `models/classifier_model_v2.pkl`, `models/recurrence_model_v2.pkl`, `models/regressor_model_v2.pkl`

