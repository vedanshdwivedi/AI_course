import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
from pathlib import Path
import pickle

# Resolve base directory relative to this script
BASE_DIR = Path(__file__).resolve().parent

model = tf.keras.models.load_model(BASE_DIR / "best_model.keras")

with open(BASE_DIR / "label_encoder_gender.pkl", "rb") as file:
    gender_encoder = pickle.load(file)

with open(BASE_DIR / "one_hot_encoder_geography.pkl", "rb") as file:
    geo_encoder = pickle.load(file)

with open(BASE_DIR / "sclaer.pkl", "rb") as file:
    scaler = pickle.load(file)


st.title("Customer Churn Prediction")

geography = st.selectbox('Geography', geo_encoder.categories_[0])
gender = st.selectbox('Gender', gender_encoder.classes_)
age = st.slider('Age', 18, 100, 25)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

def get_transformed_data(input: dict) -> dict:
    data = input.copy()
    encoded_gender, [geo_fr, geo_gr, geo_sp] = gender_encoder.transform([data["Gender"]])[0], geo_encoder.transform([[data["Geography"]]])[0]
    data["Gender"] = int(encoded_gender)
    data["Geography_France"] = int(geo_fr)
    data["Geography_Spain"] = int(geo_sp)
    data["Geography_Germany"] = int(geo_gr)
    del data["Geography"]
    return data

def get_scaled_data(input: dict):
    df = pd.DataFrame([input])
    df = df[scaler.feature_names_in_]
    scaled_array = scaler.transform(df)
    return scaled_array

def get_prediction(data):
    input = data.copy()
    new_data = get_transformed_data(input)
    new_data_scaled = get_scaled_data(new_data)
    prediction = model.predict(new_data_scaled)
    churn_probability = float(prediction[0][0])
    return churn_probability

if st.button("Predict Churn"):
    churn_prob = get_prediction(input_data)
    st.subheader(f"Churn Probability: {churn_prob:.2%}")
    if churn_prob > 0.5:
        st.error("⚠️ The customer is likely to CHURN (Exit).")
    else:
        st.success("✅ The customer is likely to STAY.")