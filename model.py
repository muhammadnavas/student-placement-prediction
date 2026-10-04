import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

import config
from preprocessing import prepare_train_test_data

def train_and_save_all_models(random_state: int = config.RANDOM_STATE):
    """
    Trains Logistic Regression, Random Forest, and Gradient Boosting models,
    extracts feature importances, and saves all models.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    data = prepare_train_test_data(random_state=random_state)
    
    X_train = data["X_train"]
    X_test = data["X_test"]
    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    
    # 1. Logistic Regression (fitted on scaled features)
    lr_model = LogisticRegression(random_state=random_state, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    with open(config.LR_MODEL_PATH, "wb") as f:
        pickle.dump(lr_model, f)
        
    # 2. Random Forest (fitted on unscaled features for robust interpretable tree splits)
    rf_model = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        random_state=random_state,
        max_depth=6,
        min_samples_leaf=2
    )
    rf_model.fit(X_train, y_train)
    with open(config.RF_MODEL_PATH, "wb") as f:
        pickle.dump(rf_model, f)
        
    # 3. Gradient Boosting (fitted on unscaled features)
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=3,
        random_state=random_state
    )
    gb_model.fit(X_train, y_train)
    with open(config.GB_MODEL_PATH, "wb") as f:
        pickle.dump(gb_model, f)
        
    # Feature Importances from Random Forest
    rf_importances = dict(zip(config.NUMERIC_FEATURES, rf_model.feature_importances_))
    sorted_importances = dict(sorted(rf_importances.items(), key=lambda item: item[1], reverse=True))
    with open(config.FEATURE_IMPORTANCE_PATH, "w") as f:
        json.dump(sorted_importances, f, indent=4)
        
    print("[+] All models trained and saved successfully.")
    return {
        "lr_model": lr_model,
        "rf_model": rf_model,
        "gb_model": gb_model,
        "feature_importances": sorted_importances,
        "data": data
    }

def get_rf_tuning_curve(n_estimators_list=[10, 25, 50, 75, 100, 150, 200, 250, 300]):
    """
    Computes accuracy and F1-score across different n_estimators values.
    """
    data = prepare_train_test_data()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    
    accuracies = []
    f1_scores = []
    
    for n in n_estimators_list:
        rf = RandomForestClassifier(n_estimators=n, random_state=config.RANDOM_STATE, max_depth=6, min_samples_leaf=2)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        
    return {
        "n_estimators": n_estimators_list,
        "accuracy": accuracies,
        "f1_score": f1_scores
    }

def load_models():
    """Loads all saved models from disk."""
    if not (os.path.exists(config.RF_MODEL_PATH) and os.path.exists(config.LR_MODEL_PATH) and os.path.exists(config.GB_MODEL_PATH)):
        train_and_save_all_models()
        
    with open(config.RF_MODEL_PATH, "rb") as f:
        rf = pickle.load(f)
    with open(config.LR_MODEL_PATH, "rb") as f:
        lr = pickle.load(f)
    with open(config.GB_MODEL_PATH, "rb") as f:
        gb = pickle.load(f)
    with open(config.FEATURE_IMPORTANCE_PATH, "r") as f:
        feat_imp = json.load(f)
        
    return {"Random Forest": rf, "Logistic Regression": lr, "Gradient Boosting": gb, "Feature Importance": feat_imp}

if __name__ == "__main__":
    train_and_save_all_models()
