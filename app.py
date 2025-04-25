import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('steam_sales_model.pkl')

# Function to preprocess input (make sure this matches your training preprocessing)
def preprocess_input(input_data):
    # Load your original scaler if you saved it during training
    # Or create a new one that matches your training preprocessing
    scaler = StandardScaler()
    # Only scale numerical features (adjust as needed)
    numerical_features = ['reviewScore', 'price']
    input_data[numerical_features] = scaler.fit_transform(input_data[numerical_features])
    return input_data

# Streamlit UI
st.title("🎮 **Steam Game Sales Predictor**")
st.write("Predict how many copies a Steam game might sell based on its features!")

# Input fields - organized in columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.header("Core Features")
    price = st.number_input("Price ($)", min_value=0.0, max_value=200.0, value=19.99, step=0.01)
    review_score = st.number_input("Review Score (0-100)", min_value=0, max_value=100, value=75)
    support_all_platforms = st.checkbox("Supports All Platforms (Windows, Mac, Linux)")

with col2:
    st.header("Publisher & Features")
    publisher_AAA = st.checkbox("Published by AAA Studio")
    publisher_AA = st.checkbox("Published by AA Studio")
    workshop_support = st.checkbox("Supports Steam Workshop")
    steam_trading_cards = st.checkbox("Has Steam Trading Cards")
    
st.header("Game Genres")
action = st.checkbox("Action Game")
other_genre = st.checkbox("Other Genre (Not Indie/Adventure/Casual/Strategy)")

# Prepare input data in the exact format of your training data
input_data = pd.DataFrame({
    'price': [price],
    'reviewScore': [review_score],
    'publisherClass_AAA': [int(publisher_AAA)],
    'publisherClass_AA': [int(publisher_AA)],
    'workshop_support': [int(workshop_support)],
    'steam_trading_cards': [int(steam_trading_cards)],
    'Action': [int(action)],
    'Others': [int(other_genre)],
    'support_all_platforms': [int(support_all_platforms)],
})

# Predict button
if st.button("Predict Sales"):
    try:
        processed_data = preprocess_input(input_data)
        prediction = model.predict(processed_data)
        
        # Display prediction with formatting
        st.success("## Prediction Result")
        st.metric(label="Predicted Copies Sold", value=f"{int(prediction[0]):,}")
        
        # Show the input data for verification
        st.subheader("Input Features Used")
        st.dataframe(input_data.style.highlight_max(axis=0))
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add some sample data for quick testing
st.sidebar.header("Quick Test Presets")
if st.sidebar.button("Load AAA Game Example"):
    st.experimental_set_query_params(
        price=59.99,
        review_score=85,
        publisher_AAA=True,
        support_all_platforms=True,
        steam_trading_cards=True
    )
    st.experimental_rerun()

if st.sidebar.button("Load Indie Game Example"):
    st.experimental_set_query_params(
        price=14.99,
        review_score=75,
        publisher_AA=True,
        action=True,
        workshop_support=True
    )
    st.experimental_rerun()
st.write('Hello, Streamlit!')

