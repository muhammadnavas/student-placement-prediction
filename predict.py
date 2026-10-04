import numpy as np
import pandas as pd
import config
from model import load_models
from preprocessing import scale_features

def predict_single_student(student_dict: dict, model_name: str = "Random Forest"):
    """
    Takes a dictionary representing one student's academic and profile data,
    validates it, formats the 8-dim vector, and returns the prediction result.
    """
    models = load_models()
    if model_name not in models:
        model_name = "Random Forest"
        
    model = models[model_name]
    
    # Construct DataFrame in exact feature order
    row_data = {feat: [float(student_dict.get(feat, 0.0))] for feat in config.NUMERIC_FEATURES}
    df_input = pd.DataFrame(row_data)
    
    if model_name == "Logistic Regression":
        X_eval = scale_features(df_input)
    else:
        X_eval = df_input
        
    pred_class = int(model.predict(X_eval)[0])
    prob_placed = float(model.predict_proba(X_eval)[0][1])
    
    if prob_placed >= 0.70:
        readiness = "High Placement Readiness"
        status_color = "success"
    elif prob_placed >= 0.45:
        readiness = "Moderate Readiness (Borderline)"
        status_color = "warning"
    else:
        readiness = "At-Risk / Needs Targeted Intervention"
        status_color = "danger"
        
    return {
        "model_used": model_name,
        "prediction": "Placed" if pred_class == 1 else "Not Placed",
        "is_placed": pred_class == 1,
        "placement_probability": round(prob_placed * 100, 2),
        "readiness_band": readiness,
        "status_color": status_color,
        "student_features": {feat: float(student_dict.get(feat, 0.0)) for feat in config.NUMERIC_FEATURES}
    }

def predict_batch_students(df_students: pd.DataFrame, model_name: str = "Random Forest"):
    """
    Runs batch inference on a DataFrame of student records.
    """
    models = load_models()
    model = models.get(model_name, models["Random Forest"])
    
    df_eval = df_students.copy()
    for feat in config.NUMERIC_FEATURES:
        if feat not in df_eval.columns:
            df_eval[feat] = 0.0
            
    X_mat = df_eval[config.NUMERIC_FEATURES]
    if model_name == "Logistic Regression":
        X_mat = scale_features(X_mat)
        
    preds = model.predict(X_mat)
    probs = model.predict_proba(X_mat)[:, 1]
    
    df_eval["Predicted_Status"] = ["Placed" if p == 1 else "Not Placed" for p in preds]
    df_eval["Placement_Probability_%"] = np.round(probs * 100, 2)
    df_eval["Readiness_Band"] = [
        "High" if p >= 0.70 else "Moderate" if p >= 0.45 else "At-Risk" for p in probs
    ]
    return df_eval
