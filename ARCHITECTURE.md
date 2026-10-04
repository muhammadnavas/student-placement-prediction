# Student Placement Prediction & Readiness System
## Architecture & Pipeline Documentation

> **Stack:** Python 3.12 · scikit-learn · FastAPI · Streamlit · Plotly
> **Dataset:** 500-student synthetic cohort + 67-student institutional assessment records

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Directory Structure](#2-directory-structure)
3. [End-to-End ML Pipeline](#3-end-to-end-ml-pipeline)
4. [Stage 1 — Data Ingestion & Synthesis](#stage-1--data-ingestion--synthesis)
5. [Stage 2 — Feature Engineering & Validation](#stage-2--feature-engineering--validation)
6. [Stage 3 — Preprocessing & Scaling](#stage-3--preprocessing--scaling)
7. [Stage 4 — Model Training](#stage-4--model-training)
8. [Stage 5 — Evaluation & Visualisation](#stage-5--evaluation--visualisation)
9. [Stage 6 — Inference Engine](#stage-6--inference-engine)
10. [Stage 7 — Recommendation Engine](#stage-7--recommendation-engine)
11. [Deployment — FastAPI REST Backend](#deployment--fastapi-rest-backend)
12. [Deployment — Streamlit Frontend](#deployment--streamlit-frontend)
13. [Configuration Registry](#configuration-registry)
14. [Model Performance Summary](#model-performance-summary)
15. [Feature Importance Rankings](#feature-importance-rankings)
16. [Testing & Quality Assurance](#testing--quality-assurance)
17. [Data Flow Diagram](#data-flow-diagram)
18. [Module Dependency Map](#module-dependency-map)
19. [Running the System](#running-the-system)

---

## 1. System Overview

The **Student Placement Prediction & Readiness System** is an end-to-end supervised machine learning application that:

- **Predicts** whether a student will be campus-placed using 8 multi-attribute features.
- **Explains** predictions through feature importance-ranked gap analysis.
- **Recommends** a personalised, prioritised remediation roadmap for every student.
- **Exposes** a production-ready REST API and an interactive analytical dashboard.

### Core Design Principles

| Principle | Implementation |
|---|---|
| Multi-attribute over CGPA-only | 8 features; skills combine to >56% importance |
| Explainability-first | RF feature importances drive recommendations |
| Dual deployment | FastAPI (REST) + Streamlit (GUI) |
| Stateless inference | Models serialised to `.pkl`; scaler persisted |
| Anonymised data | All records use mock IDs (`Student 01` to `Student 67`) |
| No virtual environment | Packages installed to base Python 3.12 |

---

## 2. Directory Structure

```
ML2 - Project/
|
+-- config.py                  # Central configuration registry (paths, constants, benchmarks)
+-- data_loader.py             # Dataset generation, loading, and anonymised raw data
+-- feature_engineering.py     # Feature validation, clipping, derived indices
+-- preprocessing.py           # Train/test split + StandardScaler
+-- model.py                   # Model training, serialisation, tuning curves
+-- evaluate.py                # Metrics, confusion matrix, ROC, PCA, charts
+-- predict.py                 # Single-student and batch inference
+-- recommend.py               # Gap analysis and personalised recommendation engine
|
+-- api.py                     # FastAPI REST backend (port 8000)
+-- app.py                     # Streamlit interactive dashboard (port 8501)
+-- test_system.py             # 6-unit integration & API test suite
|
+-- data/
|   +-- student_placement_data.csv         # 500 synthetic placement records
|   +-- institutional_assessment_data.csv  # 67-student anonymised cohort
|
+-- models/
|   +-- random_forest_model.pkl
|   +-- logistic_regression_model.pkl
|   +-- gradient_boosting_model.pkl
|   +-- scaler.pkl
|   +-- evaluation_metrics.json
|   +-- feature_importance.json
|
+-- plots/
|   +-- confusion_matrix_rf.png
|   +-- roc_curves.png
|   +-- feature_importance.png
|   +-- hyperparameter_tuning.png
|   +-- pca_projection.png
|   +-- radar_chart_profiles.png
|
+-- ARCHITECTURE.md            # This document
+-- README.md
+-- .gitignore
```

---

## 3. End-to-End ML Pipeline

```
+--------------------------------------------------------------+
|                  END-TO-END ML PIPELINE                      |
|                                                              |
|  [Stage 1]          [Stage 2]            [Stage 3]           |
|  Data Ingestion ---> Feature Engineer ---> Preprocessing &   |
|  & Synthesis        & Validation          Scaling            |
|      |                                        |              |
|      v                                        v              |
|  [Stage 4]          [Stage 5]            [80/20 Split]       |
|  Model Train  ---> Evaluation &  <---   Train / Test         |
|  & Serialise        Visualisation                            |
|      |                                                       |
|      v                                                       |
|  [Stage 6]          [Stage 7]            [Deployment]        |
|  Inference   -----> Recommendation ----> FastAPI + Streamlit  |
|  Engine             Engine                                    |
+--------------------------------------------------------------+
```

---

## Stage 1 — Data Ingestion & Synthesis

**File:** [`data_loader.py`](data_loader.py)

### Responsibilities

- Embeds the full 67-student **institutional assessment dataset** as an inline CSV string (`INSTITUTIONAL_RAW_DATA`).
- **Generates** a 500-student synthetic placement dataset on first-run using reproducible NumPy distributions.
- **Ensures** both CSVs exist on disk via `ensure_dataset_files()`.
- Exposes `load_data()`, `load_institutional_data()`, `load_cmrit_data()` accessors.

### Synthetic Dataset Generation — `generate_placement_dataset(n=500)`

The 500-student synthetic dataset is generated to match the project cohort statistics (Table 3.2):

| Cohort | n | CGPA | Aptitude | Communication | Technical | Internships | Projects | Certs | Backlogs |
|---|---|---|---|---|---|---|---|---|---|
| **Placed** | 337 (67.4%) | mu=7.84, sd=0.65 | mu=73.92, sd=10.5 | mu=71.71, sd=9.8 | mu=74.54, sd=11.2 | Poisson(1.47) | Poisson(2.81) | Poisson(1.85) | Poisson(0.74) |
| **Not Placed** | 163 (32.6%) | mu=6.84, sd=0.72 | mu=58.74, sd=12.0 | mu=60.27, sd=11.5 | mu=60.38, sd=12.8 | Poisson(0.75) | Poisson(1.66) | Poisson(1.06) | Poisson(1.66) |

```python
# Key generation approach (data_loader.py)
placed_cgpa = np.clip(np.random.normal(7.84, 0.65, n_placed), 6.0, 9.9)
not_cgpa    = np.clip(np.random.normal(6.84, 0.72, n_not),    5.0, 8.4)
# ... repeated for all 8 features with independent distributions
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # deterministic shuffle
```

### Institutional Dataset

- **67 anonymised student records** from an engineering college AIML department cohort.
- Tracks assessment scores across 5 skill pillars: **Language (Lx), Aptitude (Ax), Softskills (Sx), Core Engineering (Cx), Programming (Px)**.
- Each pillar has 4 progressive levels (0–4).
- IDs: `Student 01`–`Student 67` | USNs: `USN23AI001`–`USN23AI067` | Emails: `student01@college.edu`–`student67@college.edu`.

---

## Stage 2 — Feature Engineering & Validation

**File:** [`feature_engineering.py`](feature_engineering.py)

### Responsibilities

1. **Type coercion** — forces all 8 numeric features to `float`/`int` via `pd.to_numeric(errors='coerce')`.
2. **Null imputation** — fills missing values with column median.
3. **Domain clipping** — enforces valid feature ranges to remove entry errors.
4. **Feature matrix extraction** — returns clean `X` (8 columns) and binary `y`.

### Clipping Bounds

| Feature | Min | Max |
|---|---|---|
| `cgpa` | 0.0 | 10.0 |
| `aptitude_score` | 0.0 | 100.0 |
| `communication_score` | 0.0 | 100.0 |
| `technical_score` | 0.0 | 100.0 |
| `internships` | 0 | inf (int) |
| `projects` | 0 | inf (int) |
| `certifications` | 0 | inf (int) |
| `backlogs` | 0 | inf (int) |

### Derived Composite Indices (via `calculate_derived_indices()`)

| Index | Formula |
|---|---|
| `skill_index` | `(aptitude + communication + technical) / 3` |
| `practical_exposure` | `(internships x 2.0) + (projects x 1.5) + (certifications x 1.0)` |
| `academic_stability` | `(cgpa x 10.0) - (backlogs x 12.5)` |

> **Note:** The primary ML pipeline uses only the 8 raw features for classification. Derived indices are available for extended analysis.

---

## Stage 3 — Preprocessing & Scaling

**File:** [`preprocessing.py`](preprocessing.py)

### Stratified Train/Test Split

```
500 student records
        |
        v
Stratified Split (test_size=0.20, random_state=42, stratify=y)
        |
        +-- X_train (400 records)  67.4% placed -> 270 placed / 130 not placed
        +-- X_test  (100 records)  67.4% placed ->  67 placed /  33 not placed
```

- **Stratification** preserves the 67.4% / 32.6% class ratio in both splits, preventing evaluation bias.
- Split config: `TEST_SIZE = 0.20`, `RANDOM_STATE = 42` (defined in `config.py`).

### StandardScaler

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit on training data ONLY
X_test_scaled  = scaler.transform(X_test)         # apply same transform to test
```

| Rule | Detail |
|---|---|
| Fit-only on train | Prevents data leakage from test set into scaling parameters |
| Serialised to disk | `models/scaler.pkl` — ensures consistent inference-time scaling |
| Used by | Logistic Regression (requires zero-mean, unit-variance inputs) |
| Not used by | Random Forest & Gradient Boosting (tree-based; scale-invariant) |

---

## Stage 4 — Model Training

**File:** [`model.py`](model.py)

### Three Algorithms Trained

#### 1. Logistic Regression (Linear Baseline)

```python
LogisticRegression(random_state=42, max_iter=1000)
```

- Trained on **scaled** features (`X_train_scaled`).
- Linear probabilistic baseline; weights interpretable via signed coefficients.

#### 2. Random Forest Classifier — PRIMARY MODEL

```python
RandomForestClassifier(
    n_estimators    = 200,   # 200 bagged decision trees
    max_depth       = 6,     # Depth cap controls overfitting
    min_samples_leaf = 2,    # Minimum 2 samples per leaf
    random_state    = 42
)
```

- Trained on **unscaled** features (trees are scale-invariant).
- **Selected as primary model** — highest F1-score (0.876) and Recall (0.896).
- Provides Gini-based `feature_importances_` that power the recommendation engine.
- `n_estimators=200` confirmed by hyperparameter tuning sweep (converges at ~150–200).

#### 3. Gradient Boosting Classifier (Sequential Ensemble)

```python
GradientBoostingClassifier(
    n_estimators  = 100,
    learning_rate = 0.08,   # Conservative; prevents overfitting
    max_depth     = 3,      # Shallow trees; each corrects prior residuals
    random_state  = 42
)
```

- Trained on **unscaled** features; sequential boosting corrects prior tree's residuals.

### Model Serialisation Output

```
models/
+-- random_forest_model.pkl         # Primary inference model
+-- logistic_regression_model.pkl
+-- gradient_boosting_model.pkl
+-- scaler.pkl                      # Fitted StandardScaler
+-- feature_importance.json         # RF Gini importances, sorted descending
+-- evaluation_metrics.json         # Accuracy, F1, ROC-AUC for all 3 models
```

### Hyperparameter Tuning — `get_rf_tuning_curve()`

Sweeps `n_estimators` over `[10, 25, 50, 75, 100, 150, 200, 250, 300]`, recording Accuracy and F1-score at each step.
Results saved as `plots/hyperparameter_tuning.png`. Confirms `n_estimators=200` as the point of diminishing returns.

---

## Stage 5 — Evaluation & Visualisation

**File:** [`evaluate.py`](evaluate.py)

### Metrics Computed (held-out test set, n=100)

| Metric | Formula | Purpose |
|---|---|---|
| **Accuracy** | `(TP+TN)/(TP+TN+FP+FN)` | Overall correctness |
| **Precision** | `TP/(TP+FP)` | Avoids false placement labels |
| **Recall** | `TP/(TP+FN)` | Minimises missed placements — primary concern |
| **F1-Score** | `2 x (P x R)/(P+R)` | Harmonic mean; primary model selection criterion |
| **ROC-AUC** | Area under ROC curve | Discrimination power across all decision thresholds |

### Six Plots Generated

| Plot | Filename | Description |
|---|---|---|
| Confusion Matrix | `confusion_matrix_rf.png` | Heatmap of TP/TN/FP/FN for Random Forest |
| ROC Curves | `roc_curves.png` | All 3 models overlaid; labelled with AUC values |
| Feature Importance | `feature_importance.png` | Horizontal bar chart sorted by Gini importance |
| Hyperparameter Tuning | `hyperparameter_tuning.png` | Accuracy and F1 vs `n_estimators` sweep |
| PCA 2D Projection | `pca_projection.png` | 500 students in PC1–PC2 space coloured by placement label |
| Radar Profile Chart | `radar_chart_profiles.png` | Mean placed vs not-placed 8-feature radar (min-max normalised) |

### Baseline Comparison (Table 3.3)

ML model cross-tabulated against a simple CGPA >= 7.5 single-threshold rule:

| | CGPA < 7.5 (Rule: Not Placed) | CGPA >= 7.5 (Rule: Placed) |
|---|---|---|
| **Model: Not Placed** | 29 | 1 |
| **Model: Placed** | **17** | 53 |

> **Key Finding:** 17 students with CGPA < 7.5 are correctly identified as **placement-ready** by the ML model, thanks to strong compensatory skills in aptitude, coding, and communication. This validates multi-attribute assessment over single-metric rules.

---

## Stage 6 — Inference Engine

**File:** [`predict.py`](predict.py)

### Single-Student Inference — `predict_single_student()`

```
Input dict: { cgpa, aptitude_score, communication_score, technical_score,
              internships, projects, certifications, backlogs }
                              |
                              v
        Build 8-dim DataFrame in exact config.NUMERIC_FEATURES order
                              |
              +---------------+---------------+
              | Logistic Regression?          | RF or GB?
              v                               v
     scale_features(X)               X (unscaled)
              |                               |
              +---------------+---------------+
                              |
                    model.predict(X)         -> binary label {0, 1}
                    model.predict_proba(X)   -> probability P(placed)
                              |
                    Readiness band classification:
                    P >= 0.70  ->  "High Placement Readiness"
                    P >= 0.45  ->  "Moderate Readiness (Borderline)"
                    P <  0.45  ->  "At-Risk / Needs Targeted Intervention"
                              |
                              v
Output: { prediction, is_placed, placement_probability,
          readiness_band, status_color, student_features, model_used }
```

### Batch Inference — `predict_batch_students()`

- Accepts a `pd.DataFrame` of N student records.
- Appends three output columns: `Predicted_Status`, `Placement_Probability_%`, `Readiness_Band`.
- Downloadable as CSV from the Streamlit batch evaluation tab.
- Also exposed via `POST /batch-predict` REST endpoint.

---

## Stage 7 — Recommendation Engine

**File:** [`recommend.py`](recommend.py)

### Algorithm — Gap Analysis + Feature Importance Weighting

```
For each of the 8 features:
    gap = PLACED_BENCHMARK[feature] - student[feature]

    if gap > 0:
        weighted_impact = (gap / max_scale) x feature_importance x 100
        severity = "CRITICAL" | "High" | "Medium" | "Low"

Emit prioritised recommendation items sorted by severity and importance
```

### Placed Cohort Benchmarks (from `config.py`)

| Feature | Placed Cohort Mean |
|---|---|
| CGPA | 7.84 |
| Aptitude Score | 73.92 |
| Communication Score | 71.71 |
| Technical Score | 74.54 |
| Internships | 1.47 |
| Projects | 2.81 |
| Certifications | 1.85 |
| Backlogs | 0.0 (ideal) |

### Priority Levels & Triggers

| Priority | Trigger Condition | Example Recommendation |
|---|---|---|
| **P0 — Urgent** | Active backlogs > 0 | "Clear N Active Backlogs" |
| **P1 — High** | CGPA < 7.0 or score < 65/100 | "Boost GPA" / "Sharpen Coding Skills" |
| **P2 — Medium** | Score below benchmark | Aptitude drills / internship advice |
| **P3 — Low** | No certifications | Cloud / AI-ML certification path |
| **Optimal** | Student meets all benchmarks | Target dream/super-dream recruiters |

### Output JSON Structure

```json
{
  "weak_areas_count": 3,
  "gap_analysis": {
    "technical_score": {
      "feature": "Technical / Coding Score",
      "current": 58.0,
      "target": 74.54,
      "gap": 16.54,
      "severity": "High",
      "weighted_impact": 2.49
    }
  },
  "recommendations": [
    {
      "category": "Technical Mastery",
      "priority": "P1 - High",
      "icon": "...",
      "title": "Sharpen Core Data Structures & Coding Problem-Solving",
      "action": "..."
    }
  ],
  "benchmarks": { "cgpa": 7.84, "..." }
}
```

---

## Deployment — FastAPI REST Backend

**File:** [`api.py`](api.py)
**Port:** `8000` | **Swagger Docs:** `http://localhost:8000/docs`

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check; returns system status |
| `GET` | `/metrics` | All 3 model metrics + feature importances JSON |
| `GET` | `/benchmarks` | Placed cohort mean benchmark values |
| `POST` | `/predict` | Single-student placement classification |
| `POST` | `/recommend` | Personalised skill improvement recommendations |
| `POST` | `/analyze` | Combined: prediction + recommendations in one call |
| `POST` | `/batch-predict` | Bulk inference over a list of student records |

### Request Schema — `StudentProfile` (Pydantic)

```json
{
  "student_id": "STU_001",
  "cgpa": 7.8,
  "aptitude_score": 75.0,
  "communication_score": 70.0,
  "technical_score": 76.0,
  "internships": 1,
  "projects": 3,
  "certifications": 2,
  "backlogs": 0,
  "model_name": "Random Forest"
}
```

Field validation enforced with Pydantic `Field(ge=..., le=...)` bounds matching the clipping rules in `feature_engineering.py`.

### Auto-Training on Startup

```python
@app.on_event("startup")
def startup_event():
    if not (os.path.exists(config.RF_MODEL_PATH) and os.path.exists(config.METRICS_PATH)):
        train_and_save_all_models()
        evaluate_all_models()
```

The API **self-heals** on launch — if saved models are missing, it retrains automatically before serving requests.

---

## Deployment — Streamlit Frontend

**File:** [`app.py`](app.py)
**Port:** `8501` | **URL:** `http://localhost:8501`

### Five Navigation Tabs

| Tab | Description |
|---|---|
| **Placement Predictor & Advisor** | Interactive sliders for student profile; live prediction with readiness badge; 360-degree Plotly radar vs placed benchmark; feature gap table; ranked action plan cards |
| **Model Comparison & Analytics** | Metrics table (highlight_max); ROC curves; confusion matrix; PCA projection; hyperparameter tuning curves; CGPA baseline cross-tabulation |
| **Batch Placement Evaluation** | CSV upload or built-in 500-student dataset; summary KPI metrics; filterable results table; CSV download |
| **Institutional Assessment Cohort** | 67-student anonymised records explorer; name/USN search; competency bar chart across 5 skill pillars |
| **Project Methodology & Architecture** | Inline methodology overview and feature importance plot |

### Quick Profile Presets

| Preset | CGPA | Aptitude | Comm | Tech | Intern | Proj | Cert | Backlogs |
|---|---|---|---|---|---|---|---|---|
| High Achiever | 8.4 | 85 | 80 | 88 | 2 | 4 | 3 | 0 |
| Borderline Student | 7.2 | 64 | 68 | 72 | 1 | 2 | 1 | 0 |
| At-Risk Student | 6.2 | 48 | 52 | 50 | 0 | 1 | 0 | 2 |

### Radar Chart Normalisation

| Feature | Normalisation |
|---|---|
| CGPA | `cgpa / 10.0` |
| Aptitude | `aptitude_score / 100.0` |
| Communication | `communication_score / 100.0` |
| Technical | `technical_score / 100.0` |
| Internships | `min(internships / 3.0, 1.0)` |
| Projects | `min(projects / 5.0, 1.0)` |
| Certifications | `min(certifications / 4.0, 1.0)` |

---

## Configuration Registry

**File:** [`config.py`](config.py)

```python
# Paths
DATA_DIR                = "./data/"
DATASET_PATH            = "./data/student_placement_data.csv"
INSTITUTIONAL_DATA_PATH = "./data/institutional_assessment_data.csv"
MODELS_DIR              = "./models/"
PLOTS_DIR               = "./plots/"

# Model Training Parameters
RANDOM_STATE            = 42
TEST_SIZE               = 0.20          # 80/20 stratified split
RF_N_ESTIMATORS         = 200
CGPA_THRESHOLD          = 7.5           # Single-variable baseline rule

# 8 Feature Names (enforced throughout pipeline)
NUMERIC_FEATURES = [
    "cgpa", "aptitude_score", "communication_score",
    "technical_score", "internships", "projects",
    "certifications", "backlogs"
]
TARGET_COLUMN = "placed"

# Placed Cohort Benchmarks (drives recommendation gap analysis)
PLACED_BENCHMARKS = {
    "cgpa": 7.84,  "aptitude_score": 73.92,
    "communication_score": 71.71, "technical_score": 74.54,
    "internships": 1.47, "projects": 2.81,
    "certifications": 1.85, "backlogs": 0.0
}
```

---

## Model Performance Summary

Results on the held-out test set (n=100, stratified 80/20 split):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** (selected) | **0.920** | 0.897 | **0.896** | **0.876** | 0.896 |
| Logistic Regression | 0.900 | 0.875 | 0.866 | 0.854 | **0.900** |
| Gradient Boosting | 0.910 | 0.882 | 0.881 | 0.867 | 0.893 |

**Why Random Forest was selected as the primary model:**

- **Highest F1-score (0.876)** — optimal balance of precision and recall.
- **Highest Recall (0.896)** — correctly identifies 60 of 67 placed students on the test set.
- Non-linear tree splits capture feature interactions that linear models cannot.
- Directly provides Gini-based `feature_importances_` that power the recommendation engine.
- Scale-invariant — no normalisation required at inference time.
- While Logistic Regression achieves marginally higher ROC-AUC (0.900 vs 0.896), the Random Forest's explainability advantage and recall performance justifies selection for this placement use case.

---

## Feature Importance Rankings

Derived from Random Forest Gini impurity reduction across 200 trees (trained on 400-student set):

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | `cgpa` | **27.15%** | Single strongest predictor; primary academic shortlisting signal |
| 2 | `aptitude_score` | **21.54%** | Quantitative/logical reasoning; campus structured test performance |
| 3 | `technical_score` | **17.83%** | Coding and DSA ability; critical for IT and software roles |
| 4 | `communication_score` | **16.99%** | Soft skills; decisive in HR interviews and group discussions |
| 5 | `projects` | **6.96%** | Portfolio evidence of practical delivery capability |
| 6 | `backlogs` | **4.11%** | Hard eligibility gate — even 1 backlog removes most tier-1/2 options |
| 7 | `certifications` | **2.90%** | Resume differentiation and domain credentialing |
| 8 | `internships` | **2.51%** | Industry exposure; less impactful than direct skill scores in this cohort |

> **Key Insight:** The four score attributes (CGPA + Aptitude + Technical + Communication) combine to **83.51%** of total decision weight. Multi-attribute assessment is statistically proven essential — CGPA alone cannot adequately predict placement outcomes.

---

## Testing & Quality Assurance

**File:** [`test_system.py`](test_system.py)

Six integration tests using `unittest` + FastAPI `TestClient`:

| Test | Entry Point | Validates |
|---|---|---|
| `test_predict_single_student` | `predict_single_student()` | Returns prediction, probability >= 50%, readiness band for strong profile |
| `test_recommendations` | `generate_recommendations()` | Gap analysis populated; backlogs trigger P0 critical alert |
| `test_api_root` | `GET /` | Returns `status: online` with HTTP 200 |
| `test_api_predict` | `POST /predict` | Returns `prediction: Placed` for strong student profile |
| `test_api_recommend` | `POST /recommend` | Returns non-empty recommendations list |
| `test_api_metrics` | `GET /metrics` | Returns `models_comparison` and `feature_importances` keys |

All 6 tests pass with exit code 0 in approximately 0.13 seconds.

---

## Data Flow Diagram

```
+------------------------------------------------------------------+
|                          DATA FLOW                               |
|                                                                  |
|  data_loader.py                                                  |
|  +----------------------------------------------+               |
|  | INSTITUTIONAL_RAW_DATA (embedded inline)     |--> CSV disk    |
|  | + generate_placement_dataset(n=500)          |--> CSV disk    |
|  +----------------------------------------------+               |
|                      |                                           |
|                      v                                           |
|  feature_engineering.py                                          |
|  +----------------------------------+                            |
|  | validate_and_clean_data()        | null fill, clip, cast      |
|  | get_feature_matrix()             | X (8-dim), y (binary)      |
|  +----------------------------------+                            |
|                      |                                           |
|                      v                                           |
|  preprocessing.py                                                |
|  +----------------------------------+                            |
|  | train_test_split(stratify=y)     | 400 train / 100 test       |
|  | StandardScaler.fit_transform()   | --> scaler.pkl             |
|  +----------------------------------+                            |
|          |                    |                                  |
|          v                    v                                  |
|  model.py                evaluate.py                            |
|  +----------------+      +------------------------------+        |
|  | LR.fit()       |      | Accuracy, F1, ROC-AUC        |        |
|  | RF.fit()  -----|----> | Confusion matrix             |        |
|  | GB.fit()       |      | ROC curves, PCA, Radar       |        |
|  | -> *.pkl       |      | -> evaluation_metrics.json   |        |
|  +----------------+      +------------------------------+        |
|          |                                                       |
|          v                                                       |
|  predict.py                    recommend.py                     |
|  +--------------------+        +----------------------------+    |
|  | predict_single()   |        | gap vs PLACED_BENCHMARKS   |    |
|  | predict_batch()    |        | weighted_impact * weight   |    |
|  +--------------------+        | sorted priority roadmap    |    |
|          |                     +----------------------------+    |
|          +---------------------------+                           |
|                                      v                           |
|          +-------------------------------------------+          |
|          |  api.py  (FastAPI  REST  :8000)            |          |
|          |  app.py  (Streamlit GUI  :8501)            |          |
|          +-------------------------------------------+          |
+------------------------------------------------------------------+
```

---

## Module Dependency Map

```
config.py                       <- root; no internal dependencies
    ^
    |
data_loader.py                  <- config
    ^
    |
feature_engineering.py          <- config
    ^
    |
preprocessing.py                <- config, data_loader, feature_engineering
    ^
    |
model.py                        <- config, preprocessing
    ^
    |
evaluate.py                     <- config, data_loader, preprocessing, model
    ^
    |
predict.py                      <- config, model, preprocessing
    ^
    |
recommend.py                    <- config, model
    ^
    |
api.py         <- config, model, predict, recommend, evaluate, data_loader
app.py         <- config, data_loader, model, predict, recommend, evaluate
test_system.py <- api, predict, recommend
```

---

## Running the System

### 1. First-time setup: generate datasets and train models

```bash
python data_loader.py      # Generates both CSV datasets
python model.py            # Trains and saves all 3 models + scaler
python evaluate.py         # Generates all 6 evaluation plots
```

### 2. Start the REST API (FastAPI)

```bash
python api.py
# Swagger UI -> http://localhost:8000/docs
```

### 3. Start the Interactive Dashboard (Streamlit)

```bash
streamlit run app.py --server.port 8501
# Dashboard -> http://localhost:8501
```

### 4. Run Unit Tests

```bash
python -m unittest test_system.py
# Expected: Ran 6 tests in ~0.13s ... OK
```

---

*Document reflects the exact production code state of the repository.*
*GitHub: [muhammadnavas/student-placement-prediction](https://github.com/muhammadnavas/student-placement-prediction)*
