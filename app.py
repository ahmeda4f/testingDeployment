import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load model and scaler
# Load the trained model
# Load model, scalers, and expected features
model = joblib.load('steam_sales_model.pkl')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
feature_names = joblib.load('feature_names.pkl')  # This is new

# UI Setup
st.set_page_config(page_title="Steam Sales Predictor", layout="wide", page_icon="🎮")

# Title
st.title("🎮 Steam Game Sales Predictor")

# Input Section
col1, col2 = st.columns(2)

with col1:
    st.header("💰 Pricing & Reviews")
    price = st.number_input("Game Price ($)", min_value=0.0, max_value=200.0, value=19.99, step=0.01)
    review_score = st.number_input("Review Score (0-100)", min_value=0, max_value=100, value=75)

with col2:
    st.header("🏢 Publisher Class")
    publisher_class = st.radio("Publisher Type:", ["Indie", "AA Studio", "AAA Studio"], index=0)
    
    st.header("🛠️ Features")
    workshop = st.checkbox("Workshop Support")
    trading_cards = st.checkbox("Steam Trading Cards")

# Additional Features
with st.expander("➕ More Options"):
    st.header("🎮 Genres")
    action = st.checkbox("Action Genre")
    other_genre = st.checkbox("Other Genre")
    
    st.header("🖥️ Platforms")
    all_platforms = st.checkbox("Supports All Platforms")

# Prediction Function
def predict_sales(input_df):
    # Scale the numeric features
    input_scaled = scaler_X.transform(input_df)

    # Reindex columns to match the training set
    input_data = pd.DataFrame(input_scaled, columns=feature_names).reindex(columns=feature_names, fill_value=0)

    # Make the prediction
    prediction_scaled = model.predict(input_data)

    # Inverse scale the prediction to get actual sales values
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))

    return prediction

# Prepare input data when the button is clicked
if st.button("🚀 Predict Sales"):
    # Prepare input data
    input_df = pd.DataFrame({
        'price': [price],
        'reviewScore': [review_score],
        'publisherClass_AAA': [1 if publisher_class == "AAA Studio" else 0],
        'publisherClass_AA': [1 if publisher_class == "AA Studio" else 0],
        'workshop_support': [int(workshop)],
        'steam_trading_cards': [int(trading_cards)],
        'Action': [int(action)],
        'Others': [int(other_genre)],
        'support_all_platforms': [int(all_platforms)]
    })
    
    # Debug: Show raw input
    st.write("Raw Input Data:", input_df)
    
    # Call prediction function
    prediction = predict_sales(input_df)
    
    if prediction is not None:
        st.success(f"## Predicted Sales: {int(prediction[0][0]):,} copies")
        
        # Show feature importance (example values - replace with your model's)
        st.subheader("Top Influencing Factors")
        st.markdown(""" 
        1. Review Score (35%) 
        2. Price (25%) 
        3. AAA Publisher Status (15%) 
        4. Steam Trading Cards (10%) 
        5. Workshop Support (8%) 
        """)

# Sample Presets
st.sidebar.header("Quick Presets")
if st.sidebar.button("Indie Game Example"):
    st.session_state.price = 14.99
    st.session_state.review_score = 80
    st.session_state.publisher_class = "Indie"
    st.session_state.workshop = True
    st.session_state.trading_cards = False
    st.rerun()

if st.sidebar.button("AAA Game Example"):
    st.session_state.price = 59.99
    st.session_state.review_score = 85
    st.session_state.publisher_class = "AAA Studio"
    st.session_state.workshop = True
    st.session_state.trading_cards = True
    st.rerun()

# Model Info
st.sidebar.header("Model Details")
st.sidebar.markdown("""
- **Model Type**: Gradient Boosting
- **Training R²**: 0.85
- **Features**: 9 total
- **Best Predictors**: Review score, price
""")
