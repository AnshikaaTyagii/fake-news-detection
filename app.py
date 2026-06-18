import streamlit as st
import joblib

# Load model
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.set_page_config(page_title="Fake News Detector")

st.title("📰 Fake News Detection System")

news = st.text_area("Paste News Article Here")

if st.button("Check News"):

    if news.strip() == "":
        st.warning("Please enter some news text")
    else:
        vector = vectorizer.transform([news])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")