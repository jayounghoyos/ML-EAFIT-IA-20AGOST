import streamlit as st
import pandas as pd
import numpy as np

# Generate a 30x7 table with random data
np.random.seed(42)
data = np.random.rand(30, 7)
columns = [f"Feature_{i+1}" for i in range(7)]
df = pd.DataFrame(data, columns=columns)

st.title("Regression Model Dashboard")
st.write("Sample 30x7 Table for Model Input/Output")
st.dataframe(df)