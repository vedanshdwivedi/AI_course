import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, List

@st.cache_data
def load_data() -> Tuple[pd.DataFrame, List[str]]:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()
model = RandomForestClassifier()

model.fit(df.iloc[:,:-1],df["species"])

st.sidebar.title("Input Features")
sepal_length = st.sidebar.slider("Sepal Length", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
sepal_width = st.sidebar.slider("Sepal Width", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
petal_length = st.sidebar.slider("Petal Length", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
petal_width = st.sidebar.slider("Petal Width", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
predicition = model.predict(input_data)

st.title("Prediction")

if predicition[0] == 0:
    st.subheader("The flower is Setosa")
elif predicition[0] == 1:
    st.subheader("The flower is Versicolor")
else:
    st.subheader("The flower is Virginica")