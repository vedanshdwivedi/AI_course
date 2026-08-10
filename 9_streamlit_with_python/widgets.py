import streamlit as st
import pandas as pd

st.title("Streamlit Text Input")

name = st.text_input("Enter your name:")

age = st.slider("Select your age:", 0, 100, 25)

options = ["Python", "JavaScript", "Java", "C++"]
choice = st.selectbox("Select your favorite programming language:", options)
st.write(f"You selected: {choice}")

if name:
    st.write(f"Hello, {name}!")
    
if age:
    st.write(f"You are {age} years old.")

data = {
    "name": ["John", "Alice", "Bob"],
    "age": [25, 30, 22],
    "language": ["Python", "JavaScript", "Java"]
}

df = pd.DataFrame(data)
st.write("Here is a simple dataframe:")
st.dataframe(df)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    st.write("Here is the uploaded CSV file:")
    st.dataframe(uploaded_df)