import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load model and scalers
model = joblib.load('steam_sales_model.pkl')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
feature_names = joblib.load('feature_names.pkl')

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
    try:
        # Ensure the input has all expected features in correct order
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        
        # Scale the features
        input_scaled = scaler_X.transform(input_df)
        
        # Make prediction
        prediction_scaled = model.predict(input_scaled)
        
        # Inverse transform the prediction
        prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
        
        return prediction[0][0]  # Return single prediction value
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

# Prediction Button
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
    
  
    # Ensure all expected features are present
    missing_features = set(feature_names) - set(input_df.columns)
    if missing_features:
        st.warning(f"Missing features: {missing_features}")
        for feature in missing_features:
            input_df[feature] = 0  # Add missing features with default value
    
    # Reorder columns to match expected feature order
    input_df = input_df[feature_names]
    
    prediction = predict_sales(input_df)
    
    if prediction is not None:
        st.success(f"## Predicted Sales: {int(prediction):,} copies")
        
       # Show feature importance
st.subheader("Top Influencing Factors")
st.markdown("""
**Game Success Drivers:**  
Here's what impacts sales the most based on our analysis:

1. **AAA Publisher Status** - 17.8%  
   Being published by a major studio gives the biggest boost

2. **Workshop Support** - 11.1%  
   Modding capabilities significantly increase engagement

3. **Steam Trading Cards** - 10.4%  
   Collectible items provide noticeable lift

4. **AA Publisher Status** - 6.0%  
   Mid-sized studios still have an advantage over indies

5. **Action Genre** - 3.5%  
   Action games perform slightly better than average

6. **Other Genres** - 2.5%  
   Niche genres have smaller but measurable impact

7. **Review Score** - 2.3%  
   Quality matters, but less than publisher factors

8. **Multiplatform Support** - 2.2%  
   Supporting all platforms helps reach wider audience

9. **Price Point** - 1.6%  
   Pricing has surprisingly small direct impact
""")


# Model Info
st.sidebar.header("Model Details")
st.sidebar.markdown("""
- **Model Type**: Gradient Boosting
- **Training R²**: 0.52
- **Features**: 9 total
- **Best Predictors**:
 publisherClass_AAA,workshop_support
""")

# Debug information
if st.sidebar.checkbox("Show debug info"):
    st.sidebar.write("Feature names:", feature_names)
    st.sidebar.write("Model features:", model.feature_names_in_)
