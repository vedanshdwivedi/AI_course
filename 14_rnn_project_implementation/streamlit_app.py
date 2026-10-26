import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path

# Resolve base directory relative to this script
BASE_DIR = Path(__file__).resolve().parent
MAX_LENGTH = 200

# Load model and tokenizer
@st.cache_resource
def load_artifacts():
    model = load_model(BASE_DIR / "imdb_review_classification.keras")
    with open(BASE_DIR / "tokenizer.pickle", "rb") as file:
        tokenizer = pickle.load(file)
    return model, tokenizer

model, tokenizer = load_artifacts()

# App UI
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review below to predict whether the sentiment is Positive or Negative.")

# Text input
review_text = st.text_area("Movie Review", height=150, placeholder="Write your movie review here...")

def predict_sentiment(text: str):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)
    score = float(prediction[0][0])
    sentiment = "Positive" if score >= 0.5 else "Negative"
    return sentiment, score

# Predict button
if st.button("Predict Sentiment"):
    if review_text.strip():
        sentiment, score = predict_sentiment(review_text)
        st.subheader(f"Sentiment: {sentiment}")
        st.write(f"Confidence Score: {score:.2%}")
        
        if sentiment == "Positive":
            st.success("🎉 Positive Sentiment")
        else:
            st.error("⚠️ Negative Sentiment")
    else:
        st.warning("Please enter a review before predicting.")