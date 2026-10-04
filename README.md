# 🎓 Student Placement Prediction & Readiness System

An end-to-end Machine Learning system developed for **Project 1: Student Placement Prediction System**.

The system predicts student placement opportunities based on academic performance, aptitude test scores, communication abilities, technical coding proficiency, internship experience, project work, certifications, and active backlogs. It includes comparative classification modeling, explainability analysis, a weighted recommendation engine, a **FastAPI backend REST API**, and an interactive **Streamlit frontend dashboard**.

---

## 🌟 Key Features

1. **Supervised ML Classification Pipeline**:
   - **Logistic Regression**: Linear baseline with direct odds-ratio interpretation.
   - **Random Forest Classifier (200 Trees)**: Robust non-linear bagged ensemble achieving top F1-Score (0.876) and Recall (0.896).
   - **Gradient Boosting Classifier**: Sequential error-correcting boosted trees.
2. **Feature Importance & Multi-Attribute Insights**:
   - Demonstrates that **CGPA contributes only ~26.8%**, while **Aptitude, Communication, and Technical skills together contribute >50%**.
   - Captures compensating strengths for borderline students that simple CGPA cut-offs miss.
3. **Personalized Skill Recommendation Engine**:
   - Conducts individual gap analysis against placed cohort benchmarks.
   - Employs Random Forest feature importance weights to generate prioritized, actionable remediation plans.
4. **FastAPI Backend (`api.py`)**:
   - REST API endpoints for single prediction, batch prediction, gap analysis, metrics, and cohort benchmarks.
   - Interactive Swagger API documentation at `http://127.0.0.1:8000/docs`.
5. **Streamlit Modern Web Dashboard (`app.py`)**:
   - **🎯 Placement Predictor & Advisor**: Real-time sliders, gauge indicators, interactive 360° competency radar chart vs cohort mean, and prioritized action plan.
   - **📊 Model Comparison & Analytics**: ROC curves, Confusion Matrix, Hyperparameter tuning curves, PCA 2D clustering, and CGPA threshold cross-tabulation.
   - **📁 Batch Placement Evaluation**: CSV file upload and bulk scoring with downloadable reports.
   - **🏫 Institutional Assessment Cohort**: Exploration of the institutional assessment cohort across Language ($L_x$), Aptitude ($A_x$), Softskills ($S_x$), Core ($C_x$), and Programming ($P_x$) skill pillars.
   - **📑 Project Methodology & Architecture**: Complete project documentation and pipeline specifications.

---

## 📁 Repository Structure

```
ML2 - Project/
├── api.py                    # FastAPI REST API backend
├── app.py                    # Streamlit interactive dashboard
├── config.py                 # Configuration settings, paths & benchmarks
├── data_loader.py            # Dataset generation & institutional loader
├── feature_engineering.py    # Data validation & feature extraction
├── preprocessing.py          # Train/test split & StandardScaler
├── model.py                  # Model training, hyperparameter tuning & serialization
├── evaluate.py               # Evaluation metrics, ROC & plotting routines
├── predict.py                # Single & batch prediction inference
├── recommend.py              # Gap analysis & recommendation engine
├── test_system.py            # Comprehensive unit test suite
├── data/
│   ├── student_placement_data.csv
│   └── institutional_assessment_data.csv
├── models/
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── scaler.pkl
│   ├── evaluation_metrics.json
│   └── feature_importance.json
└── plots/
    ├── confusion_matrix_rf.png
    ├── feature_importance.png
    ├── hyperparameter_tuning.png
    ├── pca_projection.png
    ├── radar_chart_profiles.png
    └── roc_curves.png
```

---

## 🚀 How to Run

### 1. Run Unit Tests & Train Pipeline
```bash
python data_loader.py
python model.py
python evaluate.py
python test_system.py
```

### 2. Launch Streamlit Web Dashboard
```bash
streamlit run app.py
```

### 3. Launch FastAPI Backend Service (Optional)
```bash
uvicorn api:app --reload --port 8000
```
Swagger Documentation available at: `http://127.0.0.1:8000/docs`
