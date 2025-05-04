import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Scholarship Predictor", layout="wide")
st.markdown("<h1 style='text-align: center;'>🎓 Smart Scholarship Predictor</h1>", unsafe_allow_html=True)

# Load model and features
try:
    model = joblib.load("scholarship_model.pkl")
    features = joblib.load("model_features.pkl")
except FileNotFoundError:
    st.error("🚫 Required model files not found. Please check 'scholarship_model.pkl' and 'model_features.pkl'.")
    st.stop()

# ----------------------------------
# 🔍 Define Eligibility Logic
# ----------------------------------
def check_eligibility(name, perc, fee, income, is_disabled):
    if not name:
        return "⚠️ Please enter your name.", False

    min_required_percentage = 75
    if is_disabled:
        min_required_percentage -= 5  # Relaxed by 5%

    if perc < min_required_percentage:
        return f"❌ Not eligible: Percentage below {min_required_percentage}%.", False
    if fee <= 0.1 * income:
        return "❌ Not eligible: Annual fee is under 10% of income.", False
    return "✅ Eligible for prediction.", True

# ----------------------------------
# 📋 Sidebar Inputs
# ----------------------------------
st.sidebar.title("📌 Fill Student Details")

student_name = st.sidebar.text_input("👤 Student Name")
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
physically_challenged = st.sidebar.selectbox("Physically Challenged", ["Yes", "No"])
is_disabled = physically_challenged == "Yes"

academic_class = st.sidebar.selectbox("Academic Class", ["10th", "12th", "UG", "PG"])
study_year = st.sidebar.selectbox("Current Academic Year", ["2021", "2022", "2023", "2024"])
domicile_state = st.sidebar.selectbox("Domicile State", ["Uttar Pradesh", "Bihar", "Delhi", "Maharashtra", "Other"])
annual_income = st.sidebar.number_input("💰 Annual Family Income (₹)", min_value=0)
annual_fee = st.sidebar.number_input("🏫 Annual School/College Fee (₹)", min_value=0)
percentage = st.sidebar.number_input("📊 Previous Class Percentage (%)", min_value=0.0, max_value=100.0)

# ----------------------------------
# 🎯 Predict Button
# ----------------------------------
if st.sidebar.button("🎯 Predict Scholarship Eligibility"):
    msg, eligible = check_eligibility(student_name, percentage, annual_fee, annual_income, is_disabled)
    st.subheader("📢 Result Summary")
    st.info(f"""
    👤 **Student**: {student_name}  
    📈 **Percentage**: {percentage}%  
    ♿ **Physically Challenged**: {physically_challenged}  
    💰 **Income**: ₹{annual_income:,}  
    🏫 **Fee**: ₹{annual_fee:,}  
    """)
    st.markdown("---")
    st.write(msg)

    if eligible:
        # Prepare input for model
        input_dict = {
            f'gender_{gender}': 1,
            f'physically_challenged_{physically_challenged}': 1,
            f'academic_class_{academic_class}': 1,
            f'year_{study_year}': 1,
            f'domicile_state_{domicile_state}': 1,
        }
        input_df = pd.DataFrame([input_dict])
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[features]

        result = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if result == 1:
            st.success(f"🎉 Congratulations {student_name}, you're likely to receive a scholarship!")
            st.metric(label="Model Confidence", value=f"{prob*100:.2f}%")
            st.markdown("[📝 Apply Now](https://www.buddy4study.com/)")
        else:
            st.warning("⚠️ Based on prediction, you may not receive a scholarship.")
            st.metric(label="Model Confidence", value=f"{prob*100:.2f}%")

# ----------------------------------
# ℹ️ Footer
# ----------------------------------
st.markdown("---")
st.caption("Built with ❤️ by Shivanshu Sharma • Streamlit Scholarship App • 2025")
