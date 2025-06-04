import streamlit as st
import re
import joblib

# Load the model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review below:")

user_input = st.text_area("📝 Your Review")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Simple cleaning (same as training)
        cleaned = user_input.lower()
        cleaned = re.sub(r'<.*?>', '', cleaned)
        cleaned = re.sub(r'http\S+', '', cleaned)
        cleaned = re.sub(r'[^a-zA-Z]', ' ', cleaned)

        # Transform and predict
        input_vec = vectorizer.transform([cleaned])
        prediction = model.predict(input_vec)[0]

        if prediction == 1:
            st.success("✅ Positive Review")
        else:
            st.error("❌ Negative Review")