import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    layout="wide"
)

st.title("🏠 House Price Prediction System")
st.caption("XGBoost model with robust feature engineering")

# ---------------------------------
# LOAD MODEL & METADATA
# ---------------------------------
@st.cache_resource
def load_artifacts():
    with open("final_optimized_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("city_means.pkl", "rb") as f:
        city_means = pickle.load(f)

    return model, city_means

model, city_means = load_artifacts()
global_city_mean = city_means.mean()

# ---------------------------------
# FEATURE ENGINEERING
# ---------------------------------
def final_engineer(X, city_means):
    X = X.copy()

    X["sqft_living_log"] = np.log1p(X["sqft_living"])
    X["house_age"] = 2025 - X["yr_built"]
    X["luxury_score"] = X["view"] + X["condition"] + X["waterfront"]
    X["size_quality"] = X["sqft_living_log"] * (X["condition"] + 1)
    X["city_val"] = X["city"].map(city_means).fillna(global_city_mean)

    drop_cols = [
        "city", "street", "statezip", "date", "country",
        "yr_built", "yr_renovated", "waterfront",
        "sqft_living", "sqft_lot", "sqft_above", "sqft_basement"
    ]

    return X.drop(columns=[c for c in drop_cols if c in X.columns])

# ---------------------------------
# SIDEBAR INPUTS
# ---------------------------------
st.sidebar.header("🔧 Property Details")

bedrooms = st.sidebar.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.number_input("Bathrooms", 1.0, 8.0, 2.0)
floors = st.sidebar.number_input("Floors", 1, 3, 1)
sqft_living = st.sidebar.number_input("Living Area (sqft)", 400, 10000, 2100)
condition = st.sidebar.slider("Condition", 1, 5, 3)
view = st.sidebar.slider("View Quality", 0, 4, 0)
waterfront = st.sidebar.selectbox("Waterfront", ["No", "Yes"])
yr_built = st.sidebar.number_input("Year Built", 1900, 2025, 2012)
city = st.sidebar.selectbox("City", sorted(city_means.index.tolist()))

waterfront = 1 if waterfront == "Yes" else 0

# ---------------------------------
# INPUT DATAFRAME
# ---------------------------------
raw_input = pd.DataFrame({
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "floors": [floors],
    "sqft_living": [sqft_living],
    "sqft_lot": [1],
    "sqft_above": [1],
    "sqft_basement": [0],
    "view": [view],
    "condition": [condition],
    "waterfront": [waterfront],
    "yr_built": [yr_built],
    "yr_renovated": [0],
    "street": ["na"],
    "city": [city],
    "statezip": ["na"],
    "country": ["na"],
    "date": ["na"]
})

# ---------------------------------
# PREDICTION
# ---------------------------------
st.markdown("### 🔮 Prediction")

if st.button("Predict House Price"):
    X_final = final_engineer(raw_input, city_means)
    log_pred = model.predict(X_final)[0]
    price = np.expm1(log_pred)

    # Confidence band (±8%)
    low = price * 0.92
    high = price * 1.08

    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated Price", f"${price:,.0f}")
    col2.metric("Lower Bound", f"${low:,.0f}")
    col3.metric("Upper Bound", f"${high:,.0f}")

    st.info("Price range reflects mo
