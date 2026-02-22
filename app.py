import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. LOAD MODEL AND DATA ---
@st.cache_resource
def load_model():
    model = joblib.load('final_model.pkl')
    return model

@st.cache_data
def load_data():
    # Use the local copy of the dataset
    df = pd.read_csv('roblox_games_data.csv')
    return df

model = load_model()
df_raw = load_data()

# Extract unique values for categorical features from the raw data
unique_genres = df_raw['Genre'].dropna().unique()
unique_age_recommendations = df_raw['AgeRecommendation'].dropna().unique()

# --- 2. STREAMLIT APP LAYOUT ---
st.set_page_config(page_title="Roblox Game Predictor", layout="centered")
st.title("Roblox Game Success Predictor")
st.markdown("##### Predict if a Roblox game will be a 'success' (top 20% most active) based on its features.")

st.write("---")
st.header("Game Features Input")

# --- 3. INPUT WIDGETS ---

# Numeric features
game_age = st.number_input("Game Age (days since creation)", min_value=0, max_value=4000, value=365, step=1)
favorite_rate = st.number_input("Favorite Rate (Favorites / Visits)", min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.4f")
engagement_rate = st.number_input("Engagement Rate ((Likes - Dislikes) / (Likes + Dislikes))", min_value=-1.0, max_value=1.0, value=0.75, step=0.01, format="%.2f")

# Categorical features
genre = st.selectbox("Genre", options=np.insert(unique_genres, 0, '')) # Add empty string for NaN possibility
age_recommendation = st.selectbox("Age Recommendation", options=unique_age_recommendations)

st.write("---")

# --- 4. PREDICTION LOGIC ---
if st.button("Predict Success"):    
    # Create a DataFrame from inputs, matching the structure used for training
    input_data = pd.DataFrame([[game_age, favorite_rate, engagement_rate, genre, age_recommendation]],
                              columns=['game_age', 'favorite_rate', 'engagement_rate', 'Genre', 'AgeRecommendation'])
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.header("Prediction Result")
    if prediction[0] == 1:
        st.success("This game is predicted to be a **SUCCESS**!")
        st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
    else:
        st.info("This game is predicted to be **NOT A SUCCESS**.")
        st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")
        
    st.write("\n--- Note: 'Success' is defined as being in the top 20% of active games.\n---")
