import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scalers
model = joblib.load('steam_sales_model.pkl')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Get the feature names the model expects (from training)
feature_names = [
    'price',
    'reviewScore',
    'publisherClass_AAA',
    'publisherClass_AA',
    'workshop_support',
    'steam_trading_cards',
    'Action',
    'Others',
    'support_all_platforms'
]

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
def predict_sales(input_data):
    # Ensure correct column order and names
    input_data = input_data.reindex(columns=feature_names)
    
    # Scale only the numeric features
    input_data[['price', 'reviewScore']] = scaler_X.transform(input_data[['price', 'reviewScore']])
    
    # Predict and inverse scale
    prediction_scaled = model.predict(input_data)
    prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
    
    return prediction[0][0]  # Return single value

# Prediction Button
if st.button("🚀 Predict Sales"):
    # Prepare input data with EXACT feature names
    input_df = pd.DataFrame({
        'price': [price],
        'reviewScore': [review_score],
        'publisherClass_AAA': [1 if publisher_class == "AAA Studio" else 0],
        'publisherClass_AA': [1 if publisher_class == "AA Studio" else 0],
        'workshop_support': [int(workshop)],
        'steam_trading_cards': [int(trading_cards)],  # Note: matches feature_names spelling
        'Action': [int(action)],
        'Others': [int(other_genre)],
        'support_all_platforms': [int(all_platforms)]
    })
    
    # Debug: Show raw input
    st.write("Input Data Before Processing:", input_df)
    
    try:
        prediction = predict_sales(input_df)
        st.success(f"## Predicted Sales: {int(prediction):,} copies")
        
        # Show feature importance
        st.subheader("Top Influencing Factors")
        st.markdown("""
        1. Review Score
        2. Price
        3. AAA Publisher Status
        4. Steam Trading Cards
        5. Workshop Support
        """)
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.write("Current input DataFrame columns:", input_df.columns.tolist())
        st.write("Expected feature names:", feature_names)

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
