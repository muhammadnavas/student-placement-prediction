import unittest
from fastapi.testclient import TestClient
from api import app
from predict import predict_single_student
from recommend import generate_recommendations

class TestPlacementSystem(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.sample_student = {
            "student_id": "TEST_001",
            "cgpa": 7.8,
            "aptitude_score": 75.0,
            "communication_score": 70.0,
            "technical_score": 76.0,
            "internships": 1,
            "projects": 3,
            "certifications": 2,
            "backlogs": 0
        }

    def test_predict_single_student(self):
        result = predict_single_student(self.sample_student, model_name="Random Forest")
        self.assertIn("prediction", result)
        self.assertIn("placement_probability", result)
        self.assertIn("readiness_band", result)
        self.assertEqual(result["prediction"], "Placed")
        self.assertGreaterEqual(result["placement_probability"], 50.0)

    def test_recommendations(self):
        at_risk_student = {
            "student_id": "TEST_AT_RISK",
            "cgpa": 6.2,
            "aptitude_score": 45.0,
            "communication_score": 50.0,
            "technical_score": 48.0,
            "internships": 0,
            "projects": 0,
            "certifications": 0,
            "backlogs": 2
        }
        recs = generate_recommendations(at_risk_student)
        self.assertIn("recommendations", recs)
        self.assertIn("gap_analysis", recs)
        self.assertGreater(len(recs["recommendations"]), 0)
        # Check backlogs critical priority
        has_backlog_rec = any("Backlog" in r["title"] for r in recs["recommendations"])
        self.assertTrue(has_backlog_rec)

    def test_api_root(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "online")

    def test_api_predict(self):
        response = self.client.post("/predict", json=self.sample_student)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["prediction"], "Placed")

    def test_api_recommend(self):
        response = self.client.post("/recommend", json=self.sample_student)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("recommendations", data)

    def test_api_metrics(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("models_comparison", data)
        self.assertIn("feature_importances", data)

if __name__ == "__main__":
    unittest.main()
