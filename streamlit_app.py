import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    layout="wide",
)

# ---------------------------------------------------------
# SIDEBAR INFO SECTION
# ---------------------------------------------------------
st.sidebar.markdown("""
    <h2 style='color:#4b79a1; font-weight:700;'>📘 About This App</h2>
    <p style='font-size:15px;'>
        This dashboard predicts whether an employee is likely to 
        <b>Stay</b> or <b>Leave</b> the company based on:
        <ul>
            <li>Personal background</li>
            <li>Educational qualifications</li>
            <li>Job satisfaction & performance</li>
            <li>Workplace factors like overtime, commute, and recognition</li>
        </ul>
    </p>

    <h3 style='color:#4b79a1; font-weight:700;'>🎯 Problem Statement</h3>
    <p>
        Employee attrition is one of the biggest challenges in HR analytics.  
        Replacing an employee costs nearly <b>50–250% of their salary</b>.  
        This model helps organizations:
        <b>predict potential attrition early</b>, allowing HR teams to take
        preventive actions.
    </p>

    <h3 style='color:#4b79a1; font-weight:700;'>🧠 How It Works</h3>
    <p>
        Your inputs are processed using:
        <ul>
            <li>Binary Encoding</li>
            <li>Ordinal Encoding</li>
            <li>One-Hot Encoding</li>
            <li>Feature Scaling</li>
        </ul>
        The final prediction is made using a:
        <b>Gradient Boosting Classification Model</b> with hyperparameter tuning<br>
        achieving <b>74.5% F1-Score</b> on test data (75.4% train, 75.8% 10-fold CV)
    </p>

    <h3 style='color:#4b79a1; font-weight:700;'>🔐 Data Privacy</h3>
    <p>
        This app processes the data <b>only for prediction</b>.  
        No information is stored or shared.
    </p>

    <hr style='margin-top:20px;'>
    <p style='font-size:13px; text-align:center; color:#555;'>
        Developed for HR Analytics & Machine Learning Education 📊
    </p>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# COLORFUL TITLE
# ---------------------------------------------------------
st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
    ">
        <h1 style="color: white; font-size: 40px; font-weight: 700; margin: 0;">
            📊 Employee Attrition Prediction Dashboard
        </h1>
        <p style="color: #f0f0f0; font-size: 18px; margin-top: 10px;">
            Predict whether an employee is likely to stay or leave using HR, education & workplace information.
        </p>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD MODEL ARTIFACTS
# ---------------------------------------------------------
artifact = joblib.load("employee_attrition_model.joblib")

model = artifact["model"]
ohe = artifact["ohe"]
scaler = artifact["scaler"]
power_transformer = artifact.get("power_transformer")  # Use .get() for backward compatibility
binary_map = artifact["binary_map"]
ordinal_map = artifact["ordinal_map"]
feature_columns = artifact["feature_columns"]

# =========================================================
# SECTION 1: PERSONAL DETAILS
# =========================================================
st.markdown("### 🧍 Personal Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 65)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
with col3:
    marital = st.selectbox("Marital Status", ["Married", "Single"])

col4, col5, col6 = st.columns(3)

with col4:
    num_dependents = st.number_input("Number of Dependents", 0, 10)
with col5:
    distance_home = st.number_input("Distance from Home (km)", 0, 200)
with col6:
    remote = st.selectbox("Remote Work", ["Yes", "No"])


# =========================================================
# SECTION 2: EDUCATION DETAILS
# =========================================================
st.markdown("### 🎓 Education & Career Level")

col1, col2, col3 = st.columns(3)

with col1:
    education = st.selectbox(
        "Education Level",
        ["High School", "Associate Degree", "Bachelor’s Degree", "Master’s Degree", "PhD"]
    )
with col2:
    job_level = st.selectbox("Job Level", ["Entry", "Mid", "Senior"])
with col3:
    job_role = st.selectbox("Job Role", ["Education", "Media", "Healthcare", "Technology", "Finance"])


# =========================================================
# SECTION 3: PROFESSIONAL DETAILS
# =========================================================
st.markdown("### 🏢 Workplace & Performance Details")

col1, col2, col3 = st.columns(3)

with col1:
    years_company = st.number_input("Years at Company", 0, 40)
with col2:
    company_tenure = st.number_input("Company Tenure (years)", 0, 40)
with col3:
    num_promotions = st.number_input("Number of Promotions", 0, 10)

col4, col5, col6 = st.columns(3)

with col4:
    salary = st.number_input("Monthly Income", 1000, 300000)
with col5:
    overtime = st.selectbox("Overtime", ["Yes", "No"])
with col6:
    work_life = st.selectbox("Work-Life Balance", ["Poor", "Fair", "Good", "Excellent"])

col7, col8, col9 = st.columns(3)

with col7:
    job_sat = st.selectbox("Job Satisfaction", ["Low", "Medium", "High", "Very High"])
with col8:
    perf = st.selectbox("Performance Rating", ["Low", "Below Average", "Average", "High"])
with col9:
    company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])

col10, col11, col12 = st.columns(3)

with col10:
    company_rep = st.selectbox("Company Reputation", ["Poor", "Fair", "Good", "Excellent"])
with col11:
    emp_rec = st.selectbox("Employee Recognition", ["Low", "Medium", "High", "Very High"])
with col12:
    leadership = st.selectbox("Leadership Opportunities", ["Yes", "No"])

col13, col14, col15 = st.columns(3)

with col13:
    innovation = st.selectbox("Innovation Opportunities", ["Yes", "No"])


# =========================================================
# PREDICTION BUTTON
# =========================================================
st.markdown("### 🔍 Run Prediction")

if st.button("Predict Attrition", use_container_width=True):

    # Prepare a single input row
    data = {
        "Gender"                  : gender,
        "Job Role"                : job_role,
        "Work-Life Balance"       : work_life,
        "Job Satisfaction"        : job_sat,
        "Performance Rating"      : perf,
        "Education Level"         : education,
        "Job Level"               : job_level,
        "Company Size"            : company_size,
        "Company Reputation"      : company_rep,
        "Employee Recognition"    : emp_rec,
        "Overtime"                : overtime,
        "Marital Status"          : marital,
        "Remote Work"             : remote,
        "Leadership Opportunities": leadership,
        "Innovation Opportunities": innovation,
        "Age"                     : age,
        "Monthly Income"          : salary,
        "Years at Company"        : years_company,
        "Number of Promotions"    : num_promotions,
        "Distance from Home"      : distance_home,
        "Number of Dependents"    : num_dependents,
        "Company Tenure"          : company_tenure
    }

    df = pd.DataFrame([data])

    # Handle missing values before encoding
    # Fill NaN with appropriate defaults
    if df.isnull().any().any():
        st.warning("⚠️ Some values are missing. Using default values...")
        # For numeric columns, use median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median() if not df[col].isnull().all() else 0, inplace=True)
        # For categorical columns, use mode or a default value
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if not df[col].isnull().all() else 'Unknown', inplace=True)

    # Binary Encoding
    for col, mapping in binary_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            # Handle any NaN from mapping
            if df[col].isnull().any():
                df[col].fillna(0, inplace=True)

    # Ordinal Encoding
    for col, mapping in ordinal_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            # Handle any NaN from mapping (unmapped categories)
            if df[col].isnull().any():
                df[col].fillna(0, inplace=True)

    # One-Hot Encoding
    ohe_cols = ["Job Role", "Marital Status"]
    ohe_df = pd.DataFrame(ohe.transform(df[ohe_cols]), columns=ohe.get_feature_names_out(ohe_cols))
    df = df.drop(columns=ohe_cols)
    df = pd.concat([df, ohe_df], axis=1)

    # Outlier capping
    df['Years at Company'] = df['Years at Company'].apply(lambda x: 40 if x > 40 else x)

    # Check for NaN values before power transformation
    if df[['Number of Dependents', 'Number of Promotions', 'Years at Company']].isnull().any().any():
        st.error("⚠️ Error: Some numeric values are missing. Please check your inputs.")
        st.stop()

    # Skewness transformation using saved transformer
    cols_to_transform = ['Number of Dependents', 'Number of Promotions', 'Years at Company']
    df[cols_to_transform] = power_transformer.transform(df[cols_to_transform])

    # Feature engineering
    df["Opportunities"] = df["Leadership Opportunities"] + df["Innovation Opportunities"]
    df = df.drop(columns=['Leadership Opportunities', 'Innovation Opportunities'])

    # Align columns with training features
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Final NaN check before scaling
    if df.isnull().any().any():
        st.error("⚠️ Error: NaN values detected after preprocessing!")
        st.write("Columns with NaN values:")
        st.write(df.columns[df.isnull().any()].tolist())
        st.write("Data preview:")
        st.dataframe(df)
        st.stop()

    # Scale input - preserve feature names by converting back to DataFrame
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

    # Predict
    pred = model.predict(df_scaled)[0]
    result = "🚨 Employee Will Leave" if pred == 1 else "✅ Employee Will Stay"

    # Display result
    st.markdown(f"""
        <div style="
            padding: 20px;
            background-color: {'#ffcccc' if pred==1 else '#d4edda'};
            border-radius: 10px;
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            margin-top: 20px;
        ">
            {result}
        </div>
    """, unsafe_allow_html=True)
    st.balloons()
