import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import config
from data_loader import load_data
from feature_engineering import get_feature_matrix

def prepare_train_test_data(random_state: int = config.RANDOM_STATE):
    """
    Loads dataset, extracts features, performs stratified 80/20 train-test split,
    and fits StandardScaler on numeric features.
    """
    df = load_data()
    X, y = get_feature_matrix(df, include_target=True)
    
    # Stratified split to preserve class ratio (67.4% Placed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=random_state, 
        stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=config.NUMERIC_FEATURES, 
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=config.NUMERIC_FEATURES, 
        index=X_test.index
    )
    
    # Save scaler
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(config.SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
        
    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler
    }

def load_scaler():
    """Loads fitted StandardScaler."""
    if not os.path.exists(config.SCALER_PATH):
        prepare_train_test_data()
    with open(config.SCALER_PATH, "rb") as f:
        return pickle.load(f)

def scale_features(X_df: pd.DataFrame) -> pd.DataFrame:
    """Scales input DataFrame using trained scaler."""
    scaler = load_scaler()
    return pd.DataFrame(scaler.transform(X_df), columns=config.NUMERIC_FEATURES, index=X_df.index)
