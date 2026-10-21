import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ---- Step 1: Page setup ----
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("📊 Customer Segmentation Dashboard")

# ---- Step 2: Load the segmented data ----
rfm = pd.read_csv("data/rfm_segmented.csv")

# ---- Step 3: Show segment overview ----
st.header("Segment Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customers per Segment")
    st.bar_chart(rfm["Segment"].value_counts())

with col2:
    st.subheader("Average RFM per Segment")
    st.dataframe(rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean())

# ---- Step 4: Let the user predict a new customer's segment ----
st.header("Predict a New Customer's Segment")

model = joblib.load("src/kmeans_model.pkl")
scaler = joblib.load("src/scaler.pkl")

recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
frequency = st.number_input("Frequency (number of orders)", min_value=0, value=5)
monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=1000.0)

if st.button("Predict Segment"):
    X = np.array([[recency, frequency, monetary]])
    X_scaled = scaler.transform(X)
    segment = model.predict(X_scaled)[0]
    st.success(f"This customer belongs to Segment {segment}")
    