from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import Literal

# Initialize FastAPI app
app = FastAPI(
    title="Employee Attrition Prediction API",
    description="Predict whether an employee is likely to stay or leave using ML",
    version="1.0.0"
)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts
artifact = joblib.load("employee_attrition_model.joblib")
model = artifact["model"]
ohe = artifact["ohe"]
scaler = artifact["scaler"]
power_transformer = artifact.get("power_transformer")
binary_map = artifact["binary_map"]
ordinal_map = artifact["ordinal_map"]
feature_columns = artifact["feature_columns"]


# Input schema
class EmployeeInput(BaseModel):
    age: int = Field(..., ge=18, le=65, description="Age of employee")
    gender: Literal["Male", "Female"]
    marital_status: Literal["Married", "Single"]
    num_dependents: int = Field(..., ge=0, le=10)
    distance_home: int = Field(..., ge=0, le=200, description="Distance from home in km")
    remote_work: Literal["Yes", "No"]
    education_level: Literal["High School", "Associate Degree", "Bachelor's Degree", "Master's Degree", "PhD"]
    job_level: Literal["Entry", "Mid", "Senior"]
    job_role: Literal["Education", "Media", "Healthcare", "Technology", "Finance"]
    years_company: int = Field(..., ge=0, le=40)
    company_tenure: int = Field(..., ge=0, le=40)
    num_promotions: int = Field(..., ge=0, le=10)
    monthly_income: int = Field(..., ge=1000, le=300000)
    overtime: Literal["Yes", "No"]
    work_life_balance: Literal["Poor", "Fair", "Good", "Excellent"]
    job_satisfaction: Literal["Low", "Medium", "High", "Very High"]
    performance_rating: Literal["Low", "Below Average", "Average", "High"]
    company_size: Literal["Small", "Medium", "Large"]
    company_reputation: Literal["Poor", "Fair", "Good", "Excellent"]
    employee_recognition: Literal["Low", "Medium", "High", "Very High"]
    leadership_opportunities: Literal["Yes", "No"]
    innovation_opportunities: Literal["Yes", "No"]

    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "gender": "Male",
                "marital_status": "Married",
                "num_dependents": 2,
                "distance_home": 10,
                "remote_work": "No",
                "education_level": "Bachelor's Degree",
                "job_level": "Mid",
                "job_role": "Technology",
                "years_company": 5,
                "company_tenure": 5,
                "num_promotions": 2,
                "monthly_income": 75000,
                "overtime": "Yes",
                "work_life_balance": "Good",
                "job_satisfaction": "High",
                "performance_rating": "High",
                "company_size": "Large",
                "company_reputation": "Excellent",
                "employee_recognition": "High",
                "leadership_opportunities": "Yes",
                "innovation_opportunities": "Yes"
            }
        }


# Output schema
class PredictionOutput(BaseModel):
    prediction: str
    probability_stay: float
    probability_leave: float
    risk_level: str


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api")
async def api_info():
    """API endpoint information"""
    return {
        "message": "Employee Attrition Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "predict": "/predict",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
async def predict_attrition(employee: EmployeeInput):
    """
    Predict employee attrition based on provided features
    
    Returns:
    - prediction: "Will Stay" or "Will Leave"
    - probability_stay: Probability of staying (0-1)
    - probability_leave: Probability of leaving (0-1)
    - risk_level: "Low", "Medium", or "High"
    """
    try:
        # Prepare input data
        data = {
            "Gender": employee.gender,
            "Job Role": employee.job_role,
            "Work-Life Balance": employee.work_life_balance,
            "Job Satisfaction": employee.job_satisfaction,
            "Performance Rating": employee.performance_rating,
            "Education Level": employee.education_level,
            "Job Level": employee.job_level,
            "Company Size": employee.company_size,
            "Company Reputation": employee.company_reputation,
            "Employee Recognition": employee.employee_recognition,
            "Overtime": employee.overtime,
            "Marital Status": employee.marital_status,
            "Remote Work": employee.remote_work,
            "Leadership Opportunities": employee.leadership_opportunities,
            "Innovation Opportunities": employee.innovation_opportunities,
            "Age": employee.age,
            "Monthly Income": employee.monthly_income,
            "Years at Company": employee.years_company,
            "Number of Promotions": employee.num_promotions,
            "Distance from Home": employee.distance_home,
            "Number of Dependents": employee.num_dependents,
            "Company Tenure": employee.company_tenure
        }

        df = pd.DataFrame([data])

        # Binary Encoding
        for col, mapping in binary_map.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

        # Ordinal Encoding
        for col, mapping in ordinal_map.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)

        # One-Hot Encoding
        ohe_cols = ["Job Role", "Marital Status"]
        ohe_df = pd.DataFrame(
            ohe.transform(df[ohe_cols]), 
            columns=ohe.get_feature_names_out(ohe_cols)
        )
        df = df.drop(columns=ohe_cols)
        df = pd.concat([df, ohe_df], axis=1)

        # Outlier capping
        df['Years at Company'] = df['Years at Company'].apply(lambda x: 40 if x > 40 else x)

        # Skewness transformation
        cols_to_transform = ['Number of Dependents', 'Number of Promotions', 'Years at Company']
        df[cols_to_transform] = power_transformer.transform(df[cols_to_transform])

        # Feature engineering
        df["Opportunities"] = df["Leadership Opportunities"] + df["Innovation Opportunities"]
        df = df.drop(columns=['Leadership Opportunities', 'Innovation Opportunities'])

        # Align columns
        df = df.reindex(columns=feature_columns, fill_value=0)

        # Scale input
        df_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(df_scaled)[0]
        probabilities = model.predict_proba(df_scaled)[0]

        prob_stay = float(probabilities[0])
        prob_leave = float(probabilities[1])

        # Determine risk level
        if prob_leave < 0.3:
            risk_level = "Low"
        elif prob_leave < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        result = "Will Leave" if prediction == 1 else "Will Stay"

        return PredictionOutput(
            prediction=result,
            probability_stay=prob_stay,
            probability_leave=prob_leave,
            risk_level=risk_level
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)