import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load('steam_sales_model.pkl')
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
feature_names = joblib.load('feature_names.pkl')

st.set_page_config(page_title="Steam Sales Predictor", layout="wide", page_icon="🎮")
st.title("🎮 Steam Game Sales Predictor")

col1, col2 = st.columns(2)

with col1:
    st.header("💰 Pricing & Reviews")
    price = st.number_input("Game Price ($)", min_value=0.0, max_value=200.0, value=19.99, step=0.01)
    review_score = st.number_input("Review Score (0-100)", min_value=0, max_value=100, value=75)

with col2:
    st.header("🏢 Publisher Class")
    publisher_class = st.radio("Publisher Type:", ["Indie", "AA Studio", "AAA Studio"], index=1)
    
    st.header("🛠️ Features")
    workshop = st.checkbox("Workshop Support", value=True)
    trading_cards = st.checkbox("Steam Trading Cards", value=True)

with st.expander("➕ More Options", expanded=False):
    st.header("🎮 Genres")
    action = st.checkbox("Action Genre", value=True)
    other_genre = st.checkbox("Other Genre", value=False)

    st.header("🖥️ Platforms")
    all_platforms = st.checkbox("Supports All Platforms", value=True)

def predict_sales(input_df):
    try:
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        input_scaled = scaler_X.transform(input_df)
        prediction_scaled = model.predict(input_scaled)
        prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
        return prediction[0][0]
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

if st.button("🚀 Predict Sales"):
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
    
    missing_features = set(feature_names) - set(input_df.columns)
    if missing_features:
        for feature in missing_features:
            input_df[feature] = 0
    
    input_df = input_df[feature_names]
    
    prediction = predict_sales(input_df)
    
    if prediction is not None:
        st.success(f"## Predicted Sales: {int(prediction):,} copies")
        
        st.subheader("Top Influencing Factors")
        st.markdown("""
**Game Success Drivers:**  

1. **AAA Publisher Status** - 17.8%  
2. **Workshop Support** - 11.1%  
3. **Steam Trading Cards** - 10.4%  
4. **AA Publisher Status** - 6.0%  
5. **Action Genre** - 3.5%   
""")

st.sidebar.header("Model Details")
st.sidebar.markdown("""
- **Model Type**: Gradient Boosting
- **Training R²**: 0.50
- **Features**: 10 total
- **Best Predictors**: publisherClass_AAA, workshop_support
""")
