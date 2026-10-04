import pandas as pd
import numpy as np
import config

def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates feature ranges, handles missing values, and ensures clean data types.
    """
    df = df.copy()
    
    # Fill numeric nulls with median if present
    for col in config.NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    
    # Clip to valid domains
    if "cgpa" in df.columns:
        df["cgpa"] = df["cgpa"].clip(0.0, 10.0)
    if "aptitude_score" in df.columns:
        df["aptitude_score"] = df["aptitude_score"].clip(0.0, 100.0)
    if "communication_score" in df.columns:
        df["communication_score"] = df["communication_score"].clip(0.0, 100.0)
    if "technical_score" in df.columns:
        df["technical_score"] = df["technical_score"].clip(0.0, 100.0)
    if "internships" in df.columns:
        df["internships"] = df["internships"].clip(lower=0).astype(int)
    if "projects" in df.columns:
        df["projects"] = df["projects"].clip(lower=0).astype(int)
    if "certifications" in df.columns:
        df["certifications"] = df["certifications"].clip(lower=0).astype(int)
    if "backlogs" in df.columns:
        df["backlogs"] = df["backlogs"].clip(lower=0).astype(int)
        
    return df

def calculate_derived_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates derived scores for deep student profile assessment.
    """
    df = df.copy()
    df["skill_index"] = (df["aptitude_score"] + df["communication_score"] + df["technical_score"]) / 3.0
    df["practical_exposure"] = (df["internships"] * 2.0) + (df["projects"] * 1.5) + (df["certifications"] * 1.0)
    df["academic_stability"] = (df["cgpa"] * 10.0) - (df["backlogs"] * 12.5)
    return df

def get_feature_matrix(df: pd.DataFrame, include_target: bool = True):
    """
    Extracts standard 8-dimensional feature vector X and binary target y.
    """
    df_clean = validate_and_clean_data(df)
    X = df_clean[config.NUMERIC_FEATURES]
    if include_target and config.TARGET_COLUMN in df_clean.columns:
        y = df_clean[config.TARGET_COLUMN].astype(int)
        return X, y
    return X
