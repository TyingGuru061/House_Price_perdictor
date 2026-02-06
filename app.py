import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏠 House Price Prediction App")
st.write("Predict house prices using a trained **XGBoost model**")

# -----------------------------
# LOAD MODEL & CITY MEANS
# -----------------------------
@st.cache_resource
def load_artifacts():
    with open("final_optimized_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("city_means.pkl", "rb") as f:
        city_means = pickle.load(f)

    return model, city_means

model, city_means = load_artifacts()

# -----------------------------
# FEATURE ENGINEERING FUNCTION
# -----------------------------
def final_engineer(X_df, train_means):
    X = X_df.copy()

    X["sqft_living_log"] = np.log1p(X["sqft_living"])
    X["house_age"] = 2025 - X["yr_built"]
    X["luxury_score"] = X["view"] + X["condition"] + X["waterfront"]
    X["size_quality"] = X["sqft_living_log"] * (X["condition"] + 1)
    X["city_val"] = X["city"].map(train_means).fillna(train_means.mean())

    cols_to_drop = [
        "city", "street", "statezip", "date", "country",
        "yr_built", "yr_renovated", "waterfront",
        "sqft_living", "sqft_lot", "sqft_above", "sqft_basement"
    ]

    X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])
    return X

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("🏗️ House Features")

bedrooms = st.sidebar.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.number_input("Bathrooms", 1.0, 10.0, 2.0)
floors = st.sidebar.number_input("Floors", 1, 3, 1)
sqft_living = st.sidebar.number_input("Living Area (sqft)", 300, 10000, 2100)
view = st.sidebar.slider("View Score", 0, 4, 0)
condition = st.sidebar.slider("Condition", 1, 5, 3)
waterfront = st.sidebar.selectbox("Waterfront", [0, 1])
yr_built = st.sidebar.number_input("Year Built", 1900, 2025, 2010)
city = st.sidebar.selectbox("City", sorted(city_means.index.tolist()))

# -----------------------------
# CREATE INPUT DATAFRAME
# -----------------------------
input_df = pd.DataFrame({
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "floors": [floors],
    "sqft_living": [sqft_living],
    "sqft_lot": [0],
    "sqft_above": [0],
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

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔮 Predict Price"):
    engineered_input = final_engineer(input_df, city_means)
    log_price = model.predict(engineered_input)
    price = np.expm1(log_price)[0]

    st.success(f"💰 Estimated House Price: **${price:,.2f}**")

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.header("📊 Model Feature Importance")

importance_df = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax)
ax.set_title("What Drives House Prices?")
st.pyplot(fig)
