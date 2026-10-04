import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os

import config
from data_loader import load_data, load_cmrit_data, ensure_dataset_files
from model import load_models, train_and_save_all_models
from predict import predict_single_student, predict_batch_students
from recommend import generate_recommendations
from evaluate import evaluate_all_models

# Page Configuration
st.set_page_config(
    page_title="Student Placement Prediction & Readiness System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%);
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .status-badge-placed {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .status-badge-not-placed {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .rec-card {
        background-color: #FFFFFF;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        margin-bottom: 12px;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        border-top: 1px solid #F3F4F6;
        border-right: 1px solid #F3F4F6;
        border-bottom: 1px solid #F3F4F6;
    }
    .rec-urgent {
        border-left-color: #EF4444;
        background-color: #FEF2F2;
    }
    .rec-high {
        border-left-color: #F59E0B;
    }
    .rec-medium {
        border-left-color: #3B82F6;
    }
    .rec-low {
        border-left-color: #10B981;
    }
</style>
""", unsafe_allow_html=True)

# Ensure data and models exist
@st.cache_resource
def initialize_system():
    ensure_dataset_files()
    if not (os.path.exists(config.RF_MODEL_PATH) and os.path.exists(config.METRICS_PATH)):
        train_and_save_all_models()
        evaluate_all_models()
    return True

initialize_system()

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=70)
st.sidebar.title("Placement AI System")

app_mode = st.sidebar.radio(
    "Navigation Menu",
    [
        "🎯 Placement Predictor & Advisor",
        "📊 Model Comparison & Analytics",
        "📁 Batch Placement Evaluation",
        "🏫 Institutional Assessment Cohort",
        "📑 Project Methodology & Architecture"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("System Information")
st.sidebar.info(
    "**Selected Base Model:** Random Forest Classifier (200 Trees)\n\n"
    "**Benchmark Test F1:** 0.876\n\n"
    "**ROC-AUC:** 0.896\n\n"
    "**Total Evaluated Cohort:** 500 Students"
)

# -------------------------------------------------------------
# TAB 1: INDIVIDUAL PREDICTION & ADVISOR
# -------------------------------------------------------------
if app_mode == "🎯 Placement Predictor & Advisor":
    st.markdown('<div class="main-header">🎯 Student Placement Prediction & Readiness Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Assess placement likelihood, compare skills against cohort benchmarks, and receive personalized recommendations.</div>', unsafe_allow_html=True)
    
    col_input, col_result = st.columns([1.1, 1.3], gap="large")
    
    with col_input:
        st.subheader("📝 Student Profile Input")
        
        preset = st.selectbox(
            "Quick Profile Presets:",
            ["Custom Student", "Preset A: High Achiever", "Preset B: Borderline Student", "Preset C: At-Risk Student (Low Aptitude / Backlogs)"]
        )
        
        # Default values based on presets
        if preset == "Preset A: High Achiever":
            def_id, def_cgpa, def_apt, def_comm, def_tech, def_intern, def_proj, def_cert, def_back = "STU_001", 8.4, 85.0, 80.0, 88.0, 2, 4, 3, 0
        elif preset == "Preset B: Borderline Student":
            def_id, def_cgpa, def_apt, def_comm, def_tech, def_intern, def_proj, def_cert, def_back = "STU_042", 7.2, 64.0, 68.0, 72.0, 1, 2, 1, 0
        elif preset == "Preset C: At-Risk Student (Low Aptitude / Backlogs)":
            def_id, def_cgpa, def_apt, def_comm, def_tech, def_intern, def_proj, def_cert, def_back = "STU_105", 6.2, 48.0, 52.0, 50.0, 0, 1, 0, 2
        else:
            def_id, def_cgpa, def_apt, def_comm, def_tech, def_intern, def_proj, def_cert, def_back = "STU_NEW", 7.5, 70.0, 70.0, 72.0, 1, 2, 1, 0
            
        student_id = st.text_input("Student Identifier / USN:", value=def_id)
        
        c1, c2 = st.columns(2)
        with c1:
            cgpa = st.slider("Cumulative GPA (out of 10):", min_value=0.0, max_value=10.0, value=float(def_cgpa), step=0.1)
            aptitude_score = st.slider("Aptitude Score (out of 100):", min_value=0.0, max_value=100.0, value=float(def_apt), step=1.0)
            communication_score = st.slider("Communication Score (out of 100):", min_value=0.0, max_value=100.0, value=float(def_comm), step=1.0)
            technical_score = st.slider("Technical/Coding Score (out of 100):", min_value=0.0, max_value=100.0, value=float(def_tech), step=1.0)
        with c2:
            internships = st.number_input("Internships Completed:", min_value=0, max_value=10, value=int(def_intern), step=1)
            projects = st.number_input("Academic / Personal Projects:", min_value=0, max_value=15, value=int(def_proj), step=1)
            certifications = st.number_input("Relevant Certifications:", min_value=0, max_value=10, value=int(def_cert), step=1)
            backlogs = st.number_input("Active Academic Backlogs:", min_value=0, max_value=10, value=int(def_back), step=1)
            
        model_choice = st.selectbox(
            "Classification Algorithm to Use:",
            ["Random Forest (Recommended)", "Logistic Regression", "Gradient Boosting"]
        )
        clean_model_name = model_choice.replace(" (Recommended)", "")
        
    student_dict = {
        "student_id": student_id,
        "cgpa": cgpa,
        "aptitude_score": aptitude_score,
        "communication_score": communication_score,
        "technical_score": technical_score,
        "internships": internships,
        "projects": projects,
        "certifications": certifications,
        "backlogs": backlogs
    }
    
    # Run prediction & recommendations
    pred_res = predict_single_student(student_dict, model_name=clean_model_name)
    rec_res = generate_recommendations(student_dict)
    
    with col_result:
        st.subheader("📊 Assessment Outcome & Readiness Index")
        
        p_prob = pred_res["placement_probability"]
        is_p = pred_res["is_placed"]
        
        # Status Card
        res_box_color = "#EBF5FF" if is_p else "#FFF5F5"
        st.markdown(f"""
        <div style="background-color: {res_box_color}; border-radius: 12px; padding: 22px; border: 1px solid {'#93C5FD' if is_p else '#FCA5A5'}; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h3 style="margin:0; color: #1E293B;">Prediction for {student_id}</h3>
                    <p style="margin:4px 0 0 0; color: #64748B;">Evaluated via <b>{clean_model_name}</b></p>
                </div>
                <div>
                    <span class="{'status-badge-placed' if is_p else 'status-badge-not-placed'}">
                        { '✅ ' + pred_res['prediction'].upper() if is_p else '⚠️ ' + pred_res['prediction'].upper()}
                    </span>
                </div>
            </div>
            <hr style="margin: 15px 0; border: none; border-top: 1px solid {'#BFDBFE' if is_p else '#FECACA'};"/>
            <div style="display: flex; justify-content: space-around; text-align: center;">
                <div>
                    <span style="font-size: 0.85rem; color: #64748B;">Placement Propensity</span><br/>
                    <span style="font-size: 2.0rem; font-weight: 700; color: {'#059669' if p_prob >= 70 else '#D97706' if p_prob >= 45 else '#DC2626'};">{p_prob}%</span>
                </div>
                <div>
                    <span style="font-size: 0.85rem; color: #64748B;">Readiness Classification</span><br/>
                    <span style="font-size: 1.15rem; font-weight: 600; color: #1E293B;">{pred_res['readiness_band']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 360-Degree Radar Chart vs Benchmarks
    st.divider()
    st.subheader("🕸️ 360° Competency Radar vs. Placed Cohort Benchmark")
    
    radar_col1, radar_col2 = st.columns([1.2, 1], gap="large")
    
    with radar_col1:
        # Min-max normalized scores for radar representation
        radar_cats = ["CGPA", "Aptitude", "Communication", "Technical", "Internships", "Projects", "Certifications"]
        
        # Normalization scale factors
        val_student_norm = [
            cgpa / 10.0,
            aptitude_score / 100.0,
            communication_score / 100.0,
            technical_score / 100.0,
            min(internships / 3.0, 1.0),
            min(projects / 5.0, 1.0),
            min(certifications / 4.0, 1.0)
        ]
        
        val_bench_norm = [
            config.PLACED_BENCHMARKS["cgpa"] / 10.0,
            config.PLACED_BENCHMARKS["aptitude_score"] / 100.0,
            config.PLACED_BENCHMARKS["communication_score"] / 100.0,
            config.PLACED_BENCHMARKS["technical_score"] / 100.0,
            config.PLACED_BENCHMARKS["internships"] / 3.0,
            config.PLACED_BENCHMARKS["projects"] / 5.0,
            config.PLACED_BENCHMARKS["certifications"] / 4.0
        ]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=val_student_norm + [val_student_norm[0]],
            theta=radar_cats + [radar_cats[0]],
            fill='toself',
            name=f'Student ({student_id})',
            line_color='#2563EB',
            fillcolor='rgba(37, 99, 235, 0.2)'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=val_bench_norm + [val_bench_norm[0]],
            theta=radar_cats + [radar_cats[0]],
            fill='toself',
            name='Placed Cohort Mean',
            line_color='#059669',
            fillcolor='rgba(5, 150, 105, 0.15)'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1.0])
            ),
            showlegend=True,
            height=380,
            margin=dict(l=30, r=30, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with radar_col2:
        st.write("#### 🎯 Feature Gap Breakdown")
        gap_data = []
        for feat, info in rec_res["gap_analysis"].items():
            gap_data.append({
                "Metric": info["feature"],
                "Student Value": info["current"],
                "Placed Target": info["target"],
                "Severity": info["severity"],
                "Weighted Impact": f"{info['weighted_impact']}%"
            })
        if gap_data:
            st.dataframe(pd.DataFrame(gap_data), hide_index=True, use_container_width=True)
        else:
            st.success("🌟 Outstanding Profile! The student meets or exceeds placed benchmark standards across all measured metrics.")
            
    # Recommendations List
    st.divider()
    st.subheader("💡 Personalized Career & Skill Action Plan")
    st.caption("Prioritized steps derived from Random Forest feature importance weighting and identified gap severity.")
    
    for rec in rec_res["recommendations"]:
        p_class = "rec-urgent" if "Urgent" in rec["priority"] else "rec-high" if "High" in rec["priority"] else "rec-medium"
        st.markdown(f"""
        <div class="rec-card {p_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-weight: 700; font-size: 1.05rem; color: #1E293B;">{rec['icon']} {rec['title']}</span>
                <span style="font-size: 0.8rem; font-weight: 600; padding: 2px 8px; border-radius: 4px; background: #EEF2F6; color: #334155;">{rec['priority']} · {rec['category']}</span>
            </div>
            <p style="margin: 0; color: #475569; font-size: 0.95rem; line-height: 1.5;">{rec['action']}</p>
        </div>
        """, unsafe_allow_html=True)


# -------------------------------------------------------------
# TAB 2: MODEL COMPARISON & ANALYTICS
# -------------------------------------------------------------
elif app_mode == "📊 Model Comparison & Analytics":
    st.markdown('<div class="main-header">📊 Model Evaluation & Comparative Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical validation on 100 held-out test students comparing Logistic Regression, Random Forest, and Gradient Boosting.</div>', unsafe_allow_html=True)
    
    # 1. Performance Table
    with open(config.METRICS_PATH, "r") as f:
        metrics_list = json.load(f)
    metrics_df = pd.DataFrame(metrics_list)
    
    st.subheader("🏆 Model Comparison Table (Test Set, n=100)")
    st.dataframe(
        metrics_df.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="#D1FAE5"),
        use_container_width=True
    )
    
    st.info("""
    **Analytical Justification for Random Forest Selection:**
    - **Random Forest** achieved the highest **F1-Score (0.876)** and highest **Recall (0.896)**, correctly identifying 60 out of 67 placed students on the held-out test set.
    - While **Logistic Regression** achieved a marginally higher ROC-AUC (0.900 vs 0.896), Random Forest provides robust non-linear tree splits invariant to monotonic scaling and directly provides non-linear feature importances that power the **Skill Recommendation Engine**.
    """)
    
    st.divider()
    
    # Plots Row 1
    c1, c2 = st.columns(2)
    with c1:
        st.write("#### 📈 ROC Curves Comparison")
        if os.path.exists(os.path.join(config.PLOTS_DIR, "roc_curves.png")):
            st.image(os.path.join(config.PLOTS_DIR, "roc_curves.png"), use_container_width=True)
            
    with c2:
        st.write("#### 🔲 Confusion Matrix (Random Forest)")
        if os.path.exists(os.path.join(config.PLOTS_DIR, "confusion_matrix_rf.png")):
            st.image(os.path.join(config.PLOTS_DIR, "confusion_matrix_rf.png"), use_container_width=True)
            
    # Plots Row 2
    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.write("#### 🌳 Hyperparameter Tuning (n_estimators vs Accuracy/F1)")
        if os.path.exists(os.path.join(config.PLOTS_DIR, "hyperparameter_tuning.png")):
            st.image(os.path.join(config.PLOTS_DIR, "hyperparameter_tuning.png"), use_container_width=True)
            
    with c4:
        st.write("#### 🔍 Feature Importance Ranking")
        if os.path.exists(os.path.join(config.PLOTS_DIR, "feature_importance.png")):
            st.image(os.path.join(config.PLOTS_DIR, "feature_importance.png"), use_container_width=True)
            
    # Plots Row 3
    st.divider()
    c5, c6 = st.columns(2)
    with c5:
        st.write("#### 🌌 PCA 2D Student Projection")
        if os.path.exists(os.path.join(config.PLOTS_DIR, "pca_projection.png")):
            st.image(os.path.join(config.PLOTS_DIR, "pca_projection.png"), use_container_width=True)
            
    with c6:
        st.write("#### ⚖️ ML Model vs. Simple CGPA-Only Threshold (>= 7.5)")
        cross_tab_data = {
            "Model Prediction \\ Rule": ["Not Placed (Model)", "Placed (Model)"],
            "Not Placed (CGPA < 7.5)": [29, 17],
            "Placed (CGPA >= 7.5)": [1, 53]
        }
        st.dataframe(pd.DataFrame(cross_tab_data), hide_index=True, use_container_width=True)
        st.caption("💡 **Key Finding:** 17 students with borderline CGPA (< 7.5) are accurately recognized by the ML model as placement-ready due to strong compensatory aptitude, coding, and communication skills.")


# -------------------------------------------------------------
# TAB 3: BATCH PLACEMENT EVALUATION
# -------------------------------------------------------------
elif app_mode == "📁 Batch Placement Evaluation":
    st.markdown('<div class="main-header">📁 Batch Student Placement Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a CSV file or evaluate full institutional cohorts for placement forecasting.</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Student Batch CSV (Columns: cgpa, aptitude_score, communication_score, technical_score, internships, projects, certifications, backlogs):", type=["csv"])
    
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    else:
        st.info("ℹ️ No file uploaded. You can test using the built-in 500-student placement dataset.")
        if st.button("Load 500-Student Placement Dataset"):
            batch_df = load_data()
        else:
            batch_df = load_data().head(20)
            
    st.write(f"### Evaluating {len(batch_df)} Student Records")
    
    # Run batch prediction
    results_df = predict_batch_students(batch_df, model_name="Random Forest")
    
    # Summary Metrics
    total = len(results_df)
    placed_n = (results_df["Predicted_Status"] == "Placed").sum()
    not_placed_n = (results_df["Predicted_Status"] == "Not Placed").sum()
    rate = (placed_n / total) * 100 if total > 0 else 0
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Evaluated", f"{total} Students")
    m2.metric("Predicted Placed", f"{placed_n}", f"{rate:.1f}% Rate")
    m3.metric("At-Risk / Not Placed", f"{not_placed_n}", delta_color="inverse")
    m4.metric("Avg Placement Probability", f"{results_df['Placement_Probability_%'].mean():.1f}%")
    
    st.divider()
    
    # Filter controls
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        status_filter = st.multiselect("Filter by Predicted Status:", ["Placed", "Not Placed"], default=["Placed", "Not Placed"])
    with f_col2:
        band_filter = st.multiselect("Filter by Readiness Band:", ["High", "Moderate", "At-Risk"], default=["High", "Moderate", "At-Risk"])
        
    filtered_df = results_df[
        (results_df["Predicted_Status"].isin(status_filter)) &
        (results_df["Readiness_Band"].isin(band_filter))
    ]
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download Button
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predicted Placement Report (CSV)",
        data=csv_bytes,
        file_name="placement_prediction_results.csv",
        mime="text/csv"
    )


# -------------------------------------------------------------
# TAB 4: INSTITUTIONAL ASSESSMENT COHORT
# -------------------------------------------------------------
elif app_mode == "🏫 Institutional Assessment Cohort":
    st.markdown('<div class="main-header">🏫 Institutional Skill Assessment Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Department-level student assessment tracks: Language (Lx), Aptitude (Ax), Softskill (Sx), Core (Cx), and Programming (Px).</div>', unsafe_allow_html=True)
    
    inst_df = load_cmrit_data()
    
    st.write(f"**Loaded {len(inst_df)} Student Records** from Engineering Cohort.")
    
    # Metric Summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cohort Size", f"{len(inst_df)} Students")
    c2.metric("Mean Aptitude Level", f"{inst_df['Ax_Level'].mean():.2f} / 4")
    c3.metric("Mean Programming Level", f"{inst_df['Px_Level'].mean():.2f} / 4")
    c4.metric("Mean Softskill Level", f"{inst_df['Sx_Level'].mean():.2f} / 4")
    
    st.divider()
    
    # Search Student
    search_query = st.text_input("🔍 Search Student by Name or USN:", "")
    if search_query:
        display_inst = inst_df[
            inst_df["Full Name"].str.contains(search_query, case=False, na=False) |
            inst_df["USN"].str.contains(search_query, case=False, na=False)
        ]
    else:
        display_inst = inst_df
        
    st.dataframe(display_inst, use_container_width=True)
    
    st.divider()
    st.subheader("📊 Competency Distribution Across Skill Pillars")
    
    skill_cols = ["Lx_Level", "Ax_Level", "Sx_Level", "Cx_Level", "Px_Level"]
    avg_levels = inst_df[skill_cols].mean().reset_index()
    avg_levels.columns = ["Pillar", "Average Level Reached (Max 4)"]
    avg_levels["Pillar"] = ["Language (Lx)", "Aptitude (Ax)", "Softskills (Sx)", "Core Engineering (Cx)", "Programming (Px)"]
    
    fig_bar = px.bar(
        avg_levels, 
        x="Pillar", 
        y="Average Level Reached (Max 4)",
        color="Average Level Reached (Max 4)",
        color_continuous_scale="Blues",
        text_auto=".2f",
        title="Mean Skill Pillar Achievement Across Cohort"
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# -------------------------------------------------------------
# TAB 5: METHODOLOGY & ARCHITECTURE
# -------------------------------------------------------------
elif app_mode == "📑 Project Methodology & Architecture":
    st.markdown('<div class="main-header">📑 Project Methodology & System Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">System Architecture and Machine Learning Pipeline Documentation.</div>', unsafe_allow_html=True)
    
    st.write("""
    ### 📌 System Overview
    - **System Title:** Student Placement Prediction & Readiness Assessment System
    - **Methodology:** Supervised Machine Learning Pipeline with Multi-Attribute Classification & Explainability
    
    ---
    
    ### 🏗️ End-to-End Pipeline Stages
    1. **Data Ingestion & Synthesis:** 500 records with 8 key predictor attributes (`cgpa`, `aptitude_score`, `communication_score`, `technical_score`, `internships`, `projects`, `certifications`, `backlogs`) and target `placed`.
    2. **Preprocessing & Scaling:** Data integrity verification, clipping out-of-domain anomalies, stratified 80/20 train/test split, zero-mean unit-variance `StandardScaler` for linear models.
    3. **Classification Algorithm Training:** Logistic Regression (linear baseline), Random Forest Classifier (ensemble bagged trees), Gradient Boosting (sequential boosting).
    4. **Evaluation & Verification:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, and cross-tabulation against simple single-variable threshold rules.
    5. **Explainability & Feature Ranking:** Determining primary drivers of placement (CGPA: 26.8%, Aptitude: 18.9%, Communication: 16.5%, Technical: 15.0%).
    6. **Personalized Recommendation Engine:** Gap quantification against placed cohort benchmarks with weighted remediation roadmaps.
    7. **Interactive Deployment:** Dual-layer architecture with FastAPI REST service backend and Streamlit analytical dashboard frontend.
    """)
    
    st.image(os.path.join(config.PLOTS_DIR, "feature_importance.png"), caption="Figure: Feature Importance for Placement Prediction", width=650)
