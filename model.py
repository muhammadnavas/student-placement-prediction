import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import config
from preprocessing import prepare_train_test_data

def train_and_save_model(random_state: int = config.RANDOM_STATE):
    """
    Trains Logistic Regression on standard-scaled features and serializes the model.
    """
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    data = prepare_train_test_data(random_state=random_state)
    
    X_train_scaled = data["X_train_scaled"]
    y_train = data["y_train"]
    
    lr_model = LogisticRegression(random_state=random_state, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump(lr_model, f)
        
    print("[+] Logistic Regression model trained and saved successfully.")
    return {
        "model": lr_model,
        "data": data
    }

train_and_save_all_models = train_and_save_model

def get_lr_tuning_curve(c_values=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]):
    """
    Computes accuracy and F1-score across different regularization parameter (C) values
    for Logistic Regression.
    """
    data = prepare_train_test_data()
    X_train_scaled, X_test_scaled = data["X_train_scaled"], data["X_test_scaled"]
    y_train, y_test = data["y_train"], data["y_test"]
    
    accuracies = []
    f1_scores = []
    
    for c in c_values:
        model = LogisticRegression(C=c, random_state=config.RANDOM_STATE, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracies.append(accuracy_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        
    return {
        "c_values": c_values,
        "accuracy": accuracies,
        "f1_score": f1_scores
    }

def load_models():
    """Loads Logistic Regression model from disk."""
    if not os.path.exists(config.MODEL_PATH):
        train_and_save_model()
        
    with open(config.MODEL_PATH, "rb") as f:
        lr = pickle.load(f)
        
    return {"Logistic Regression": lr}

if __name__ == "__main__":
    train_and_save_model()
