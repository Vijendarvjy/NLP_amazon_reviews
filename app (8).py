import streamlit as st
import pandas as pd
import pickle
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# --- Page Configuration ---
st.set_page_config(page_title="Review Insight Dashboard", page_icon="📊", layout="wide")

# --- Styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }
    h1 { color: #2e4053; }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def clean_text(text):
    text = re.sub(r'<.*?>', '', str(text))
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return " ".join(text.split())

@st.cache_resource
def load_assets():
    with open('naive_bayes_model.pkl', 'rb') as f1:
        model = pickle.load(f1)
    with open('tfidf_vectorizer.pkl', 'rb') as f2:
        vectorizer = pickle.load(f2)
    return model, vectorizer

# --- Dashboard Header ---
st.title("🌟 Amazon Review Sentiment Analyzer")
st.markdown("Analyze customer feedback with machine learning and visual dashboards.")

# --- Sidebar ---
st.sidebar.header("Configuration")
accent_color = st.sidebar.color_picker("Pick a Dashboard Accent Color", "#00FFAA")

# --- Main Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Predict Single Review")
    user_input = st.text_area("Enter your review here:", placeholder="e.g., The quality is amazing and looks exactly like the picture!")
    
    if st.button("Run Analysis"):
        if user_input:
            model, vectorizer = load_assets()
            cleaned = clean_text(user_input)
            
            # Vectorize and Predict using the loaded model
            transformed = vectorizer.transform([cleaned])
            prediction = model.predict(transformed)[0]
            
            # Also calculate sentiment for extra context
            sentiment = TextBlob(cleaned).sentiment.polarity
            
            st.metric(label="Predicted Star Rating", value=f"{prediction} ⭐")
            st.write(f"**Text Polarity (Sentiment):** {sentiment:.2f}")
            
            if prediction >= 4:
                st.success("Positive Prediction")
            elif prediction <= 2:
                st.error("Negative Prediction")
            else:
                st.warning("Neutral Prediction")

with col2:
    st.subheader("📊 Insight Dashboard")
    chart_data = pd.DataFrame({
        'Metric': ['Quality', 'Delivery', 'Price', 'Visuals'],
        'Score': [85, 70, 90, 65]
    })
    
    fig, ax = plt.subplots()
    sns.barplot(x='Metric', y='Score', data=chart_data, palette="viridis", ax=ax)
    ax.set_title("Product Performance Metrics")
    st.pyplot(fig)

# --- Footer ---
st.divider()
st.info("Model and Vectorizer loaded from local pickle files.")
