import streamlit as st
import pickle
import numpy as np

# Load TF-IDF vectorizer and Logistic Regression model
@st.cache_resource
def load_model():
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("lr_model.pkl", "rb") as f:
        lr = pickle.load(f)
    return tfidf, lr

tfidf, lr = load_model()
feature_names = np.array(tfidf.get_feature_names_out())
coef = lr.coef_[0]

def get_word_importance(review_text):
    vec = tfidf.transform([review_text])
    word_indices = np.where(vec.toarray()[0] > 0)[0]
    words = feature_names[word_indices]
    scores = vec.toarray()[0][word_indices]
    contributions = coef[word_indices] * scores
    # Top positive and negative contributions for this review
    pos_idx = np.argsort(contributions)[-10:][::-1]
    neg_idx = np.argsort(contributions)[:10]
    return words[pos_idx], words[neg_idx]

st.title("TF-IDF Sentiment Prediction & Word Importance")
review = st.text_area("Enter your product review:")

if st.button("Predict Sentiment"):
    vec = tfidf.transform([review])
    pred = lr.predict(vec)[0]
    pred_label = "Positive" if pred == "Positive" or pred == 1 else "Negative"
    
    st.markdown(f"### Sentiment: **{pred_label}**")
    
    top_pos, top_neg = get_word_importance(review)
    st.write("**Top positive words influencing prediction:**")
    st.write(", ".join(top_pos))
    st.write("**Top negative words influencing prediction:**")
    st.write(", ".join(top_neg))