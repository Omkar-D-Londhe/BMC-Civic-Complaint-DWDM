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
