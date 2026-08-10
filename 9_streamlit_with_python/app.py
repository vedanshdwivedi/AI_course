import streamlit as st
import pandas as pd
import numpy as np

# Create a title for the page
st.title("Hello Streamlit")

# Add simple texts
st.write("This is a simple text added to the page using Streamlit.")

# Create a dataframe and display it
df = pd.DataFrame({
    'Column 1': [1, 2, 3, 4],
    'Column 2': ['A', 'B', 'C', 'D']
})
st.write("Here is a simple dataframe:")
st.dataframe(df)

# We can create linecharts as well
st.write("Here is a simple line chart:")
chart_data = pd.DataFrame(
    np.random.randn(200, 3),
    columns=['a', 'b', 'c']
)
st.line_chart(chart_data)