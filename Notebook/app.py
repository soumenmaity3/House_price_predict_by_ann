import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
from difflib import get_close_matches


from rapidfuzz import process

def find_similar_localities(user_text, locality_list, limit=5, min_score=60):
    """
    Finds closest matching locality names using fuzzy matching.
    Returns a list of suggested locality names.
    """
    matches = process.extract(user_text, locality_list, limit=limit)
    # Keep only matches with score >= min_score
    suggestions = [m[0] for m in matches if m[1] >= min_score]
    return suggestions


# --- Define Custom Objects for Robust Keras Loading (Needed for .h5 files) ---
CUSTOM_OBJECTS = {
    'mse': tf.keras.losses.MeanSquaredError(),
    'mean_squared_error': tf.keras.losses.MeanSquaredError(),
}

# -------------------------
# Load Pickle Helper
# -------------------------
def load_pickle(path):
    """Loads a standard pickle file and handles file not found errors."""
    if not os.path.exists(path):
        st.error(f"❌ Missing required file: **{path}**")
        st.stop()
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading pickle file **{path}**: {e}")
        st.stop()

# -------------------------
# Load Model & Preprocessors
# -------------------------
# --- FIX APPLIED HERE: Changed extension to .h5 ---
MODEL_PATH = "model.h5" 

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found: **{MODEL_PATH}**. Please ensure it exists.")
    st.stop()

# --- Load model using custom_objects for stability ---
try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=CUSTOM_OBJECTS)
except Exception as e:
    st.error(f"❌ Error loading Keras Model **{MODEL_PATH}**: {e}")
    st.stop()


# Load the preprocessors
locality_encoder = load_pickle("lb_encoder.pkl")
onehot_encoder = load_pickle("one_encoder.pkl")
scaler = load_pickle("scaler.pkl")

# keep a list of locality names for suggestions
_localities = list(locality_encoder.classes_)

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="House Rent Predictor", layout="centered")
st.title("🏠 House Rent Prediction App")
st.markdown("Enter details below to estimate monthly rent.")
st.markdown("---")

with st.form("rent_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Property Dimensions")
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=2, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
        size = st.number_input("Size (sqft)", min_value=50, max_value=20000, value=1000, step=50)

    with col2:
        st.subheader("Floor Details")
        current_floor = st.number_input("Current Floor", min_value=-2, max_value=200, value=1, step=1)
        total_floors = st.number_input("Total Floors", min_value=1, max_value=200, value=10, step=1)

    st.markdown("---")
    st.subheader("Tenant and Area Details")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        # Tenant preference
        bachelor = st.selectbox("Bachelor Allowed?", ["No", "Yes"]) == "Yes"
        family = st.selectbox("Family Allowed?", ["Yes", "No"]) == "Yes"

    with col4:
        # One-hot categories
        area_type = st.selectbox("Area Type", onehot_encoder.categories_[0], index=0)

    with col5:
        furnishing = st.selectbox("Furnishing Status", onehot_encoder.categories_[2], index=1)
        
    # City and Locality
    city_col, locality_col = st.columns([1, 2])
    with city_col:
        city = st.selectbox("City", onehot_encoder.categories_[1], index=0)
    
    with locality_col:
        # Locality input
        locality_text = st.text_input("Locality Name (exact spelling)", "Whitefield").strip()

    st.markdown("---")
    submit = st.form_submit_button("Predict Rent", type="primary")

# -------------------------
# Input Validations (logical + realistic)
# -------------------------
def input_checks(bhk, size, bathrooms, current_floor, total_floors, locality_text):
    errors = []
    warnings = []

    # 1. Floor logic
    if current_floor > total_floors:
        errors.append("Current floor cannot be greater than total floors.")

    # 2. Size vs BHK (practical lower bound)
    # Use a conservative per-BHK minimum sqft (you can tune these numbers)
    min_sq_per_bhk = 200  # minimal unrealistic threshold; use 250 for stricter
    if size < bhk * min_sq_per_bhk:
        warnings.append(f"Size {size} sqft is small for {bhk} BHK (min suggested {bhk * min_sq_per_bhk} sqft).")

    # 3. Bathrooms reasonable relative to BHK
    if bathrooms > bhk + 3:
        warnings.append(f"Bathrooms ({bathrooms}) unusually high for {bhk} BHK.")

    if bathrooms < 1:
        errors.append("Bathrooms must be at least 1.")

    # 4. BHK reasonable maximum
    if bhk > 8:
        warnings.append("BHK > 8 is rare; ensure this is correct.")

    # 5. Extremely small or huge size
    if size < 100:
        errors.append("Size too small (must be >= 100 sqft).")
    if size > 20000:
        warnings.append("Size extremely large; ensure this is correct.")

    # 6. Locality exists? if not, we'll later propose suggestions
    if locality_text == "" or locality_text is None:
        errors.append("Locality cannot be empty.")

    return errors, warnings

# -------------------------
# Prediction Logic
# -------------------------
if submit:

    # Run checks
    errors, warnings = input_checks(bhk, size, bathrooms, current_floor, total_floors, locality_text)

    # Show warnings (but allow continue)
    for w in warnings:
        st.warning("⚠ " + w)

    # If errors, stop early
    if errors:
        for e in errors:
            st.error("❌ " + e)
        st.stop()

    # Locality handling: check if user-entered locality exists
    try:
        locality_id = locality_encoder.transform([locality_text])[0]
        locality_found = True
    except ValueError:
        locality_found = False
        st.warning(f"❌ Locality '{locality_text}' not found in training data.")

    # Suggest the closest matches using fuzzy matching
        suggestions = find_similar_localities(locality_text, _localities, limit=5)

        if suggestions:
            st.info("Did you mean one of these?")
            picked = st.selectbox(
                "Select correct locality (or keep typed one):",
                ["-- Keep typed value --"] + suggestions
            )

            if picked != "-- Keep typed value --":
                locality_text = picked
                locality_id = locality_encoder.transform([picked])[0]
                locality_found = True

        if not locality_found:
            st.error("❌ No exact locality match. Please re-type locality correctly.")
            st.stop()

    # One-hot encode categorical inputs
    onehot_input = [[area_type, city, furnishing]]
    onehot_output = onehot_encoder.transform(onehot_input) 

    onehot_df = pd.DataFrame(
        onehot_output,
        columns=onehot_encoder.get_feature_names_out(["Area Type", "City", "Furnishing Status"])
    )

    # Numeric data
    numeric = pd.DataFrame({
        "BHK": [bhk],
        "Size": [size],
        "Bathroom": [bathrooms],
        "Current_Floor": [current_floor],
        "Total_Floors": [total_floors],
        "bachelor": [int(bachelor)],
        "family": [int(family)]
    })

    numeric_cols = ["BHK", "Size", "Bathroom", "Current_Floor", "Total_Floors"]

    # Scale only numeric columns
    numeric_scaled = numeric.copy()
    numeric_scaled[numeric_cols] = scaler.transform(numeric[numeric_cols])

    # Combine all non-locality features
    other_features = pd.concat([numeric_scaled, onehot_df], axis=1)
    other_np = other_features.values

    # Locality input for embedding
    locality_np = np.array([[locality_id]])

    # ---- sensitive prediction line: keep unchanged ----
    pred_log = model.predict([locality_np, other_np], verbose=0)[0][0]
    predicted_rent = np.expm1(pred_log)
    # ----------------------------------------------------

    # -------------------------
    # Terminal Printout
    # -------------------------
    print("\n================ NEW PREDICTION REQUEST ================")
    print(f"City:               {city}")
    print(f"Locality:           {locality_text} (ID: {locality_id})")
    print(f"BHK:                {bhk}")
    print(f"Size:               {size} sqft")
    print(f"Bathrooms:          {bathrooms}")
    print(f"Current Floor:      {current_floor}")
    print(f"Total Floors:       {total_floors}")
    print(f"Bachelor Allowed:   {int(bachelor)}")
    print(f"Family Allowed:     {int(family)}")
    print(f"Area Type:          {area_type}")
    print(f"Furnishing:         {furnishing}")
    print("-------------------------------------------------------")
    print(f"💰 PREDICTED RENT: ₹ {predicted_rent:,.2f}")
    print("========================================================\n")

    # -------------------------
    # Display Result in UI
    # -------------------------
    st.success(f"### 💰 Estimated Monthly Rent: **₹ {predicted_rent:,.0f}**")
    st.markdown("*(The value is a model prediction.)*")
