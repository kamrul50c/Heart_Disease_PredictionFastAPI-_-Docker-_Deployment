# app/schemas.py
# Pydantic models defining the request/response schemas for the API.

from pydantic import BaseModel, Field

class HeartInput(BaseModel):
    # Typical Heart Disease dataset features
    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = male; 0 = female)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=0, description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., ge=0, description="Serum cholestoral in mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 true, 0 false)")
    restecg: int = Field(..., ge=0, le=2, description="Resting electrocardiographic results (0-2)")
    thalach: int = Field(..., ge=0, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1 yes, 0 no)")
    oldpeak: float = Field(..., ge=0, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of the peak exercise ST segment (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-4) colored by fluoroscopy")
    thal: int = Field(..., ge=0, description="Thalassemia (typically 1,2,3 or 3,6,7 depending on dataset)")

class PredictionOutput(BaseModel):
    heart_disease: bool
    probability: float = Field(..., ge=0.0, le=1.0, description="Model probability of heart disease (positive class)")
    