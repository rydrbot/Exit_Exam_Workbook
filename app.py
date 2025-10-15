import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load saved model and tokenizer
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('lstm_sentiment_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model()

max_len = 100  # Use the same as your training

st.title("Amazon Review Sentiment Predictor")

review = st.text_area("Enter your product review:")

if st.button("Predict Sentiment"):
    # Preprocess input
    seq = tokenizer.texts_to_sequences([review])
    pad = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(pad)[0][0]
    sentiment = "Positive" if pred > 0.5 else "Negative"
    st.markdown(f"### Sentiment: **{sentiment}**")

    st.write(f"Model confidence: {pred:.4f}")