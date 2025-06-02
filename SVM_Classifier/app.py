import streamlit as st
import joblib
import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sc = joblib.load('spam_model.pkl')# Input form
vectorizer = joblib.load('vectorizer.pkl')


st.set_page_config(page_title="Spam Detector", layout="centered")

st.title("📩 SMS Spam Classifier")
st.write("Enter a message below to check whether it's **Spam** or **Not Spam**.")

# User input
user_input = st.text_area("✍️ Enter your message here:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        # Vectorize the user input
        input_vector = vectorizer.transform([user_input])

        # Predict using the GridSearchCV model
        prediction = sc.predict(input_vector)[0]

        # Show result
        if prediction == 1:
            st.error("🚫 This message is **Spam**.")
        else:
            st.success("✅ This message is **Not Spam**.")