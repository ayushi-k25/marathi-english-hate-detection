
import streamlit as st
import joblib
import re
import unicodedata

# Load saved model and vectorizer
model = joblib.load(
    "/content/marathi_english_hate_speech_svm.pkl"
)

vectorizer = joblib.load(
    "/content/marathi_english_tfidf_vectorizer.pkl"
)

# Text preprocessing
def clean_text(text):
    text = unicodedata.normalize('NFKC', str(text))
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Page configuration
st.set_page_config(
    page_title="Marathi-English Hate Speech Detection",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Marathi-English Hate Speech Detection")

st.write(
    "Enter a Marathi-English sentence "
    "to identify its category."
)

user_input = st.text_area(
    "Enter your sentence:",
    placeholder="Type your sentence here..."
)

if st.button("Analyze Text"):

    if user_input.strip():

        cleaned = clean_text(user_input)

        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]

        st.subheader("Classification Result")

        if prediction == "HATE":
            st.error("HATE")

        elif prediction == "OFFENSIVE":
            st.warning("OFFENSIVE")

        else:
            st.success("NEUTRAL")

        st.write("Predicted category:", prediction)

        st.caption(
            "This is an automated prediction. "
            "Code-mixed language may sometimes be misclassified."
        )

    else:
        st.warning("Please enter a sentence.")
