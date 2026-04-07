import streamlit as st
import pickle
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("log_model.pkl", "rb") as f:
    log_model = pickle.load(f)

# URL cleaning function
def clean_url(url):
    url = url.lower().replace("http://", "").replace("https://", "").strip("/")
    return url

st.title("Phishing Website Detector (Logistic Regression)")

st.write("Enter a website URL below to check if it's legitimate or phishing:")

url_input = st.text_input("Enter URL:")

if st.button("Check"):
    if url_input.strip() == "":
        st.warning("Please enter a valid URL.")
    else:
        cleaned_url = clean_url(url_input)
        features = vectorizer.transform([cleaned_url])
        pred = log_model.predict(features)[0]
        prob = log_model.predict_proba(features)[0][pred]

        result = "Phishing" if pred == 1 else "Legitimate"

        if pred == 1:
            st.error(f"⚠️ The website is: {result} (Confidence: {prob:.2f})")
        else:
            st.success(f"✅ The website is: {result} (Confidence: {prob:.2f})")
