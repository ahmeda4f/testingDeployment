import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model, scalers, and feature names
try:
    model = joblib.load('steam_sales_model.pkl')
    scaler_X = joblib.load('scaler_X.pkl')  # For input features
    scaler_y = joblib.load('scaler_y.pkl')  # For target variable
    feature_names = joblib.load('feature_names.pkl')  # Load feature names used during training
except Exception as e:
    st.error(f"Failed to load model files: {str(e)}")
    st.stop()

# Streamlit UI Configuration
st.set_page_config(page_title="Steam Sales Predictor", layout="wide", page_icon="🎮")
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
    other_genre = st.checkbox("Other Genre (Not Indie/Adventure/Casual/Strategy)")
    
    st.header("🖥️ Platforms")
    all_platforms = st.checkbox("Supports All Platforms (Windows, Mac, Linux)")

def prepare_input():
    """Prepare input data matching your dataset's column order"""
    input_df = pd.DataFrame({
        'publisherClass_AAA': [1 if publisher_class == "AAA Studio" else 0],
        'workshop_support': [1 if workshop else 0],
        'steam_trading_cards': [1 if trading_cards else 0],
        'publisherClass_AA': [1 if publisher_class == "AA Studio" else 0],
        'Action': [1 if action else 0],
        'Others': [1 if other_genre else 0],
        'reviewScore': [review_score],
        'support_all_platforms': [1 if all_platforms else 0],
        'price': [price]  # Last column to match your data
    })

    # Reorder the columns based on the feature names used in training
    input_df = input_df[feature_names]
    return input_df

def predict_sales(input_df):
    """Make prediction ensuring proper feature order and scaling"""
    try:
        # 1. Scale only the numeric features (price, reviewScore)
        numeric_features = ['reviewScore', 'price']
        input_df[numeric_features] = scaler_X.transform(input_df[numeric_features])
        
        # 2. Predict and inverse transform
        prediction_scaled = model.predict(input_df)  # Model gives scaled predictions
        prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))  # Inverse transform
        return prediction[0][0]  # Return single prediction
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Current features:", input_df.columns.tolist())
        st.write("Expected features:", feature_names)
        return None

# Prediction Button
if st.button("🚀 Predict Sales"):
    with st.spinner('Making prediction...'):
        # Prepare input data
        input_df = prepare_input()
        
        # Debug view
        st.write("Input Data Preview:", input_df)
        
        # Make prediction
        prediction = predict_sales(input_df)
        
        if prediction is not None:
            st.success(f"## Predicted Sales: {int(prediction):,} copies")
            
            # Show feature importance
            st.subheader("Key Factors Affecting Sales")
            st.markdown("""
            - Review Score (most important)
            - Price
            - Publisher Type (AAA/AA/Indie)
            - Steam Trading Cards
            - Workshop Support
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

if st.sidebar.button("AA Studio Example"):
    st.session_state.price = 29.99
    st.session_state.review_score = 75
    st.session_state.publisher_class = "AA Studio"
    st.session_state.workshop = False
    st.session_state.trading_cards = True
    st.rerun()

if st.sidebar.button("AAA Blockbuster Example"):
    st.session_state.price = 59.99
    st.session_state.review_score = 85
    st.session_state.publisher_class = "AAA Studio"
    st.session_state.workshop = True
    st.session_state.trading_cards = True
    st.rerun()

# Model Info
st.sidebar.header("Model Information")
st.sidebar.markdown("""
- **Model Type**: Gradient Boosting Regressor
- **Trained On**: Steam game sales data
- **Key Features**: 
  - Price (last column)
  - Review Score
  - Publisher Class
  - Game Features
""")
