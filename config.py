import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Paths
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "student_placement_data.csv")
INSTITUTIONAL_DATA_PATH = os.path.join(DATA_DIR, "institutional_assessment_data.csv")
CMRIT_DATA_PATH = INSTITUTIONAL_DATA_PATH  # Backwards compatibility alias

# Models Directory
MODELS_DIR = os.path.join(BASE_DIR, "models")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_model.pkl")
LR_MODEL_PATH = os.path.join(MODELS_DIR, "logistic_regression_model.pkl")
GB_MODEL_PATH = os.path.join(MODELS_DIR, "gradient_boosting_model.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "evaluation_metrics.json")
FEATURE_IMPORTANCE_PATH = os.path.join(MODELS_DIR, "feature_importance.json")

# Plots Directory
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Model Training Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.20  # 80/20 split (400 train, 100 test)
RF_N_ESTIMATORS = 200

# Feature Names
NUMERIC_FEATURES = [
    "cgpa",
    "aptitude_score",
    "communication_score",
    "technical_score",
    "internships",
    "projects",
    "certifications",
    "backlogs"
]

TARGET_COLUMN = "placed"

# Threshold for simple baseline rule
CGPA_THRESHOLD = 7.5

# Cohort Placed Benchmarks (Mean values of placed cohort for recommendation gap analysis)
PLACED_BENCHMARKS = {
    "cgpa": 7.84,
    "aptitude_score": 73.92,
    "communication_score": 71.71,
    "technical_score": 74.54,
    "internships": 1.47,
    "projects": 2.81,
    "certifications": 1.85,
    "backlogs": 0.0  # ideal is 0 backlogs
}
