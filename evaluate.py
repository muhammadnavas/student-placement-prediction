import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.decomposition import PCA

import config
from data_loader import load_data
from preprocessing import prepare_train_test_data
from model import load_models, get_lr_tuning_curve, train_and_save_model

# Ensure plots directory exists
os.makedirs(config.PLOTS_DIR, exist_ok=True)

def evaluate_model():
    """
    Evaluates Logistic Regression on the 100-student test set and generates 
    exact metrics, tables, and visualization charts.
    """
    models_dict = load_models()
    lr = models_dict["Logistic Regression"]
    
    data = prepare_train_test_data()
    X_test, X_test_scaled, y_test = data["X_test"], data["X_test_scaled"], data["y_test"]
    df = load_data()
    
    # Generate Predictions & Probabilities
    y_pred = lr.predict(X_test_scaled)
    y_prob = lr.predict_proba(X_test_scaled)[:, 1]
    
    # 1. Metrics Table
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    metrics_list = [{
        "Model": "Logistic Regression",
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1-Score": round(f1, 3),
        "ROC-AUC": round(auc, 3)
    }]
    metrics_df = pd.DataFrame(metrics_list)
    
    # Save metrics JSON
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics_list, f, indent=4)
        
    # 2. Confusion Matrix for Logistic Regression
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=["Not Placed", "Placed"],
                yticklabels=["Not Placed", "Placed"],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title("Confusion Matrix — Logistic Regression (Test Set, n=100)", fontsize=12, pad=12)
    plt.xlabel("Predicted", fontsize=11)
    plt.ylabel("Actual", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(config.PLOTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    
    # 3. ROC Curve for Logistic Regression
    plt.figure(figsize=(7, 6), dpi=300)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC={auc:.3f})", color="#1f77b4", linewidth=2.5)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("ROC Curve — Logistic Regression", fontsize=12, pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(config.PLOTS_DIR, "roc_curves.png")
    plt.savefig(roc_path)
    plt.close()
    
    # 4. Hyperparameter Tuning Curves (C Regularization Parameter)
    tuning_res = get_lr_tuning_curve()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    ax1.plot(tuning_res["c_values"], tuning_res["accuracy"], marker='o', color="#1f77b4")
    ax1.set_xscale("log")
    ax1.set_title("Accuracy vs Regularization (C)", fontsize=11)
    ax1.set_xlabel("C (log scale)", fontsize=10)
    ax1.set_ylabel("Accuracy", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(tuning_res["c_values"], tuning_res["f1_score"], marker='o', color="#ff7f0e")
    ax2.set_xscale("log")
    ax2.set_title("F1-score vs Regularization (C)", fontsize=11)
    ax2.set_xlabel("C (log scale)", fontsize=10)
    ax2.set_ylabel("F1-score", fontsize=10)
    best_idx = int(np.argmax(tuning_res["f1_score"]))
    best_c = tuning_res["c_values"][best_idx]
    ax2.axvline(best_c, linestyle="--", color="gray", label=f"best C={best_c}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    tune_path = os.path.join(config.PLOTS_DIR, "hyperparameter_tuning.png")
    plt.savefig(tune_path)
    plt.close()
    
    # 6. PCA Projection 2D
    pca = PCA(n_components=2, random_state=config.RANDOM_STATE)
    X_all = df[config.NUMERIC_FEATURES]
    X_pca = pca.fit_transform(X_all)
    var_exp = pca.explained_variance_ratio_ * 100
    
    plt.figure(figsize=(7, 5.5), dpi=300)
    placed_mask = df["placed"] == 1
    plt.scatter(X_pca[placed_mask, 0], X_pca[placed_mask, 1], color="#2ca02c", alpha=0.65, label=f"Placed (n={sum(placed_mask)})", s=25)
    plt.scatter(X_pca[~placed_mask, 0], X_pca[~placed_mask, 1], color="#d62728", alpha=0.65, label=f"Not Placed (n={sum(~placed_mask)})", s=25)
    plt.title("Student Records Projected via PCA (coloured by placement outcome)", fontsize=11, pad=10)
    plt.xlabel(f"PC1 ({var_exp[0]:.1f}% variance)", fontsize=10)
    plt.ylabel(f"PC2 ({var_exp[1]:.1f}% variance)", fontsize=10)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pca_path = os.path.join(config.PLOTS_DIR, "pca_projection.png")
    plt.savefig(pca_path)
    plt.close()
    
    # 7. Radar Chart of Placed vs Not Placed profiles
    features_radar = ["cgpa", "aptitude_score", "communication_score", "technical_score", "internships", "projects", "certifications", "backlogs"]
    labels_radar = ["CGPA", "Aptitude", "Communication", "Technical", "Internships", "Projects", "Certifications", "Backlogs"]
    
    mean_placed = df[df["placed"] == 1][features_radar].mean()
    mean_not = df[df["placed"] == 0][features_radar].mean()
    
    min_vals = df[features_radar].min()
    max_vals = df[features_radar].max()
    
    norm_placed = (mean_placed - min_vals) / (max_vals - min_vals)
    norm_not = (mean_not - min_vals) / (max_vals - min_vals)
    
    angles = np.linspace(0, 2 * np.pi, len(features_radar), endpoint=False).tolist()
    angles += angles[:1]
    
    val_placed = norm_placed.tolist() + [norm_placed.iloc[0]]
    val_not = norm_not.tolist() + [norm_not.iloc[0]]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True), dpi=300)
    ax.plot(angles, val_placed, color='#2ca02c', linewidth=2, label='Placed')
    ax.fill(angles, val_placed, color='#2ca02c', alpha=0.15)
    ax.plot(angles, val_not, color='#d62728', linewidth=2, label='Not Placed')
    ax.fill(angles, val_not, color='#d62728', alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels_radar, fontsize=9)
    ax.set_ylim(0, 1.0)
    plt.title("Behavioural/Academic Profile: Placed vs Not Placed\n(min-max normalized)", fontsize=11, pad=15)
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    radar_path = os.path.join(config.PLOTS_DIR, "radar_chart_profiles.png")
    plt.savefig(radar_path)
    plt.close()
    
    # 8. Comparison with simple threshold rule
    cgpa_rule_pred = (X_test["cgpa"] >= config.CGPA_THRESHOLD).astype(int)
    
    cross_tab = pd.crosstab(
        pd.Series(y_pred, name="Model Prediction"),
        pd.Series(cgpa_rule_pred.values, name="Threshold Rule"),
        rownames=["Logistic Regression (0=Not Placed, 1=Placed)"],
        colnames=["CGPA >= 7.5 Rule (0=Fail, 1=Pass)"]
    )
    
    print("[+] Logistic Regression evaluation completed:")
    print(metrics_df)
    print("\n[+] Confusion Matrix:")
    print(cm)
    print("\n[+] Cross-tabulation (Logistic Regression vs CGPA >= 7.5 Rule):")
    print(cross_tab)
    
    return {
        "metrics_df": metrics_df,
        "confusion_matrix": cm.tolist(),
        "cross_tabulation": cross_tab.to_dict()
    }

evaluate_all_models = evaluate_model

if __name__ == "__main__":
    evaluate_model()
