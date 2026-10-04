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
from model import load_models, get_rf_tuning_curve, train_and_save_all_models

# Ensure plots directory exists
os.makedirs(config.PLOTS_DIR, exist_ok=True)

def evaluate_all_models():
    """
    Evaluates all 3 models on the 100-student test set and generates 
    exact metrics, tables, and visualization charts matching the project report.
    """
    models_dict = load_models()
    rf = models_dict["Random Forest"]
    lr = models_dict["Logistic Regression"]
    gb = models_dict["Gradient Boosting"]
    
    data = prepare_train_test_data()
    X_test, X_test_scaled, y_test = data["X_test"], data["X_test_scaled"], data["y_test"]
    df = load_data()
    
    # Generate Predictions & Probabilities
    preds = {
        "Random Forest": rf.predict(X_test),
        "Logistic Regression": lr.predict(X_test_scaled),
        "Gradient Boosting": gb.predict(X_test)
    }
    probs = {
        "Random Forest": rf.predict_proba(X_test)[:, 1],
        "Logistic Regression": lr.predict_proba(X_test_scaled)[:, 1],
        "Gradient Boosting": gb.predict_proba(X_test)[:, 1]
    }
    
    # 1. Metrics Table (Table 3.1)
    metrics_list = []
    for name in ["Random Forest", "Logistic Regression", "Gradient Boosting"]:
        y_pred = preds[name]
        y_prob = probs[name]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        metrics_list.append({
            "Model": name,
            "Accuracy": round(acc, 3),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1-Score": round(f1, 3),
            "ROC-AUC": round(auc, 3)
        })
    metrics_df = pd.DataFrame(metrics_list)
    
    # Save metrics JSON
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics_list, f, indent=4)
        
    # 2. Confusion Matrix for Random Forest (Fig 3.3)
    cm = confusion_matrix(y_test, preds["Random Forest"])
    plt.figure(figsize=(6, 5), dpi=300)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=["Not Placed", "Placed"],
                yticklabels=["Not Placed", "Placed"],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title("Confusion Matrix — Random Forest (Test Set, n=100)", fontsize=12, pad=12)
    plt.xlabel("Predicted", fontsize=11)
    plt.ylabel("Actual", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(config.PLOTS_DIR, "confusion_matrix_rf.png")
    plt.savefig(cm_path)
    plt.close()
    
    # 3. ROC Curves (Fig 3.5)
    plt.figure(figsize=(7, 6), dpi=300)
    for name, color in zip(["Logistic Regression", "Random Forest", "Gradient Boosting"], ["#1f77b4", "#2ca02c", "#ff7f0e"]):
        fpr, tpr, _ = roc_curve(y_test, probs[name])
        auc = roc_auc_score(y_test, probs[name])
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("ROC Curves — Model Comparison", fontsize=12, pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(config.PLOTS_DIR, "roc_curves.png")
    plt.savefig(roc_path)
    plt.close()
    
    # 4. Feature Importance Horizontal Bar Plot (Fig 3.6)
    feat_imp = models_dict["Feature Importance"]
    plt.figure(figsize=(8, 5), dpi=300)
    features = list(feat_imp.keys())[::-1]
    importances = [feat_imp[k] for k in features]
    readable_labels = [f.replace("_", " ").title() for f in features]
    plt.barh(readable_labels, importances, color="#4682b4", edgecolor="#1c4587")
    plt.xlabel("Relative Importance", fontsize=11)
    plt.title("Feature Importance for Placement Prediction (Random Forest)", fontsize=12, pad=12)
    for i, v in enumerate(importances):
        plt.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, color="#222")
    plt.xlim(0, max(importances) + 0.05)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    feat_path = os.path.join(config.PLOTS_DIR, "feature_importance.png")
    plt.savefig(feat_path)
    plt.close()
    
    # 5. Hyperparameter Tuning Curves (Fig 3.1)
    tuning_res = get_rf_tuning_curve()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    ax1.plot(tuning_res["n_estimators"], tuning_res["accuracy"], marker='o', color="#1f77b4")
    ax1.set_title("Accuracy vs n_estimators (Random Forest)", fontsize=11)
    ax1.set_xlabel("n_estimators", fontsize=10)
    ax1.set_ylabel("Accuracy", fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(tuning_res["n_estimators"], tuning_res["f1_score"], marker='o', color="#ff7f0e")
    ax2.set_title("F1-score vs n_estimators (Random Forest)", fontsize=11)
    ax2.set_xlabel("n_estimators", fontsize=10)
    ax2.set_ylabel("F1-score", fontsize=10)
    best_idx = int(np.argmax(tuning_res["f1_score"]))
    best_n = tuning_res["n_estimators"][best_idx]
    ax2.axvline(best_n, linestyle="--", color="gray", label=f"best n_estimators={best_n}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    tune_path = os.path.join(config.PLOTS_DIR, "hyperparameter_tuning.png")
    plt.savefig(tune_path)
    plt.close()
    
    # 6. PCA Projection 2D (Fig 3.2)
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
    
    # 7. Radar Chart of Placed vs Not Placed profiles (Fig 3.4)
    # Min-max normalize mean profiles for 360 radar view
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
    
    # 8. Comparison with simple threshold rule (Table 3.3)
    cgpa_rule_pred = (X_test["cgpa"] >= config.CGPA_THRESHOLD).astype(int)
    rf_pred = preds["Random Forest"]
    
    cross_tab = pd.crosstab(
        pd.Series(rf_pred, name="Model Prediction"),
        pd.Series(cgpa_rule_pred.values, name="Threshold Rule"),
        rownames=["Model (0=Not Placed, 1=Placed)"],
        colnames=["CGPA >= 7.5 Rule (0=Fail, 1=Pass)"]
    )
    
    print("[+] Model evaluation and comparison completed:")
    print(metrics_df)
    print("\n[+] Confusion Matrix (RF):")
    print(cm)
    print("\n[+] Cross-tabulation (RF Model vs CGPA >= 7.5 Rule):")
    print(cross_tab)
    
    return {
        "metrics_df": metrics_df,
        "confusion_matrix": cm.tolist(),
        "cross_tabulation": cross_tab.to_dict(),
        "feature_importances": feat_imp
    }

if __name__ == "__main__":
    evaluate_all_models()
