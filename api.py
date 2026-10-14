from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import json
import os

import config
from model import train_and_save_model
from predict import predict_single_student, predict_batch_students
from recommend import generate_recommendations
from evaluate import evaluate_all_models

app = FastAPI(
    title="Student Placement Prediction & Readiness API",
    description="Machine Learning API for student placement propensity scoring and personalized career recommendations powered by Logistic Regression.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StudentProfile(BaseModel):
    student_id: Optional[str] = "STU_DEMO"
    cgpa: float = Field(..., ge=0.0, le=10.0, description="Cumulative GPA out of 10")
    aptitude_score: float = Field(..., ge=0.0, le=100.0, description="Aptitude test score out of 100")
    communication_score: float = Field(..., ge=0.0, le=100.0, description="Communication/Soft skill score out of 100")
    technical_score: float = Field(..., ge=0.0, le=100.0, description="Technical/Coding score out of 100")
    internships: int = Field(0, ge=0, description="Number of internships completed")
    projects: int = Field(0, ge=0, description="Number of projects built")
    certifications: int = Field(0, ge=0, description="Number of certifications earned")
    backlogs: int = Field(0, ge=0, description="Number of active academic backlogs")

class BatchStudentRequest(BaseModel):
    students: List[StudentProfile]

@app.on_event("startup")
def startup_event():
    """Ensure model is trained and saved upon API launch."""
    if not (os.path.exists(config.MODEL_PATH) and os.path.exists(config.METRICS_PATH)):
        train_and_save_model()
        evaluate_all_models()

@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "Student Placement Prediction System",
        "model": "Logistic Regression",
        "documentation": "/docs"
    }

@app.get("/metrics")
def get_metrics():
    """Returns classification evaluation metrics for Logistic Regression."""
    if not os.path.exists(config.METRICS_PATH):
        evaluate_all_models()
    with open(config.METRICS_PATH, "r") as f:
        metrics = json.load(f)
    return {
        "evaluation_metrics": metrics,
        "model": "Logistic Regression"
    }

@app.get("/benchmarks")
def get_benchmarks():
    """Returns placed cohort mean benchmarks for comparison."""
    return config.PLACED_BENCHMARKS

@app.post("/predict")
def predict_placement(profile: StudentProfile):
    """Predicts placement likelihood and classification for a student using Logistic Regression."""
    try:
        data_dict = profile.model_dump()
        result = predict_single_student(data_dict)
        result["student_id"] = profile.student_id
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend")
def recommend_improvements(profile: StudentProfile):
    """Generates personalized, prioritized recommendations and skill gap analysis."""
    try:
        data_dict = profile.model_dump()
        return generate_recommendations(data_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
def analyze_student(profile: StudentProfile):
    """Performs full end-to-end analysis: prediction + probability + personalized recommendations."""
    try:
        data_dict = profile.model_dump()
        pred_res = predict_single_student(data_dict)
        rec_res = generate_recommendations(data_dict)
        return {
            "student_id": profile.student_id,
            "prediction_result": pred_res,
            "recommendation_result": rec_res
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
def batch_predict(batch: BatchStudentRequest):
    """Performs batch prediction over a list of student records using Logistic Regression."""
    try:
        records = [s.model_dump() for s in batch.students]
        df = pd.DataFrame(records)
        res_df = predict_batch_students(df)
        return {
            "total_students": len(res_df),
            "placed_count": int((res_df["Predicted_Status"] == "Placed").sum()),
            "not_placed_count": int((res_df["Predicted_Status"] == "Not Placed").sum()),
            "results": res_df.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
