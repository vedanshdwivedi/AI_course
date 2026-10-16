import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
from pathlib import Path
import pickle

# Page configuration
st.set_page_config(
    page_title="Customer Salary Predictor",
    page_icon="💰",
    layout="centered"
)

# Resolve base directory relative to this script
BASE_DIR = Path(__file__).resolve().parent

# Load the trained regression model, scaler, and encoders
model = tf.keras.models.load_model(BASE_DIR / "salary_estimate_model.keras")

with open(BASE_DIR / "label_encoder_gender.pkl", "rb") as file:
    gender_encoder = pickle.load(file)

with open(BASE_DIR / "one_hot_encoder_geography.pkl", "rb") as file:
    geo_encoder = pickle.load(file)

with open(BASE_DIR / "regressor_scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.title("💰 Customer Salary Estimation")
st.markdown("Predict the **Estimated Salary** of a customer using an Artificial Neural Network (ANN) regression model.")

st.write("---")

# Organized inputs into two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Details")
    geography = st.selectbox("Geography", geo_encoder.categories_[0])
    gender = st.selectbox("Gender", gender_encoder.classes_)
    age = st.slider("Age", min_value=18, max_value=100, value=35)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

with col2:
    st.subheader("🏦 Account Details")
    balance = st.number_input("Account Balance ($)", min_value=0.0, value=60000.0, step=1000.0, format="%.2f")
    tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=3)
    num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=2)
    has_cr_card = st.selectbox("Has Credit Card", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.selectbox("Is Active Member", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")

input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member
}

def get_transformed_data(input_dict: dict) -> dict:
    data = input_dict.copy()
    
    # Encode Gender
    encoded_gender = gender_encoder.transform([data["Gender"]])[0]
    data["Gender"] = int(encoded_gender)
    
    # Encode Geography with proper DataFrame to preserve feature names
    geo_df = pd.DataFrame([[data["Geography"]]], columns=["Geography"])
    geo_encoded = geo_encoder.transform(geo_df)[0]
    for col_name, val in zip(geo_encoder.get_feature_names_out(["Geography"]), geo_encoded):
        data[col_name] = float(val)
        
    del data["Geography"]
    return data

def get_scaled_data(input_dict: dict) -> np.ndarray:
    df = pd.DataFrame([input_dict])
    df = df[scaler.feature_names_in_]
    scaled_array = scaler.transform(df)
    return scaled_array

def predict_salary(data: dict) -> float:
    transformed = get_transformed_data(data)
    scaled = get_scaled_data(transformed)
    prediction = model.predict(scaled, verbose=0)
    salary_est = float(prediction[0][0])
    return salary_est

st.write("---")

if st.button("🚀 Predict Estimated Salary", use_container_width=True):
    with st.spinner("Calculating estimated salary using ANN..."):
        salary = predict_salary(input_data)
        
    st.success("✅ Prediction Complete!")
    st.metric(label="Predicted Estimated Salary", value=f"${salary:,.2f}")
    
    st.caption("ℹ️ This prediction is generated using the trained ANN regression model.")