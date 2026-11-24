
import streamlit as st
import joblib
import os
import re
import unicodedata
from collections import Counter

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Marathi-English NLP Analysis",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# 2. LOAD EXISTING SVM MODEL
# --------------------------------------------------

@st.cache_resource
def load_models():
    model = joblib.load(
        os.path.join(BASE_DIR, "marathi_english_hate_speech_svm.pkl")
    )

    vectorizer = joblib.load(
        os.path.join(BASE_DIR, "marathi_english_tfidf_vectorizer.pkl")
    )

    return model, vectorizer


model, tfidf_vectorizer = load_models()

# --------------------------------------------------
# 3. LOAD SENTIMENT ANALYZER
# --------------------------------------------------

sentiment_analyzer = SentimentIntensityAnalyzer()

# --------------------------------------------------
# 4. TEXT PREPROCESSING
# --------------------------------------------------

def clean_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# 5. SENTIMENT ANALYSIS
# --------------------------------------------------


def analyze_sentiment(text):
    text = text.lower()

    # Romanized Marathi-English sentiment words
    positive_words = [
        "chan", "chhan", "sundar", "changla",
        "changli", "mast", "bhari", "awadte",
        "happy", "good", "love", "excellent"
    ]

    negative_words = [
        "raag", "raga", "vait", "bekar",
        "nafrat", "dukh", "dukhi", "radu",
        "hate", "bad", "worst", "insult",
        "abuse", "angry", "sad", "terrible",
        "dislike", "disrespect", "apman"
    ]

    words = re.findall(r"\b[a-zA-Z]+\b", text)

    positive_count = sum(w in positive_words for w in words)
    negative_count = sum(w in negative_words for w in words)

    # Use VADER as an additional English-language signal
    vader_scores = sentiment_analyzer.polarity_scores(text)
    compound = vader_scores["compound"]

    if negative_count > positive_count:
        sentiment = "NEGATIVE"
    elif positive_count > negative_count:
        sentiment = "POSITIVE"
    else:
        if compound >= 0.05:
            sentiment = "POSITIVE"
        elif compound <= -0.05:
            sentiment = "NEGATIVE"
        else:
            sentiment = "NEUTRAL"

    return sentiment, vader_scores



# --------------------------------------------------
# 6. KEYWORD EXTRACTION
# --------------------------------------------------

STOPWORDS = {
    "the", "is", "are", "was", "were", "a", "an",
    "and", "or", "to", "of", "in", "on", "for",
    "tu", "mi", "he", "aahe", "ahe", "ahes",
    "khup", "mala", "tula", "nahi"
}


def extract_keywords(text):
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())

    useful_words = [
        word for word in words
        if word not in STOPWORDS and len(word) > 2
    ]

    counts = Counter(useful_words)

    return counts.most_common(5)


# --------------------------------------------------
# 7. TARGET GROUP EXTRACTION
# --------------------------------------------------

# Add the exact terms and aliases used by your dataset.
# These are illustrative examples, not a complete dictionary.


TARGET_TERMS = {
    "GENDER": [
        "women", "woman", "girl", "girls", "men", "boys",
        "mulgi", "mulgi", "mulinna", "mulina",
        "bai", "bayka", "striya", "purush"
    ],

    "RELIGION": [
        "hindu", "muslim", "christian", "sikh",
        "religion", "dharm", "dharma"
    ],

    "REGION": [
        "marathi", "bihari", "punjabi",
        "gujarati", "tamil", "bengali"
    ]
}


def extract_target_groups(text):
    text = text.lower()
    detected = []

    for category, terms in TARGET_TERMS.items():

        for term in terms:

            pattern = rf"\b{re.escape(term)}\b"

            if re.search(pattern, text):
                detected.append(category)
                break

    return detected if detected else ["NOT IDENTIFIED"]



# --------------------------------------------------
# 8. OPTIONAL LDA MODEL
# --------------------------------------------------

lda_model = None
count_vectorizer = None

try:
    lda_model = joblib.load(
        os.path.join(BASE_DIR, "lda_model_v2.pkl")
    )

    count_vectorizer = joblib.load(
        os.path.join(BASE_DIR, "count_vectorizer_v2.pkl")
    )

except FileNotFoundError:
    pass


def get_lda_topics():
    if lda_model is None or count_vectorizer is None:
        return []

    words = count_vectorizer.get_feature_names_out()
    topics = []

    for topic_index, topic_weights in enumerate(lda_model.components_):
        top_indices = topic_weights.argsort()[-10:][::-1]
        top_words = [words[i] for i in top_indices]

        topics.append({
            "topic": f"Topic {topic_index + 1}",
            "words": ", ".join(top_words)
        })

    return topics


def predict_document_topic(text):
    if lda_model is None or count_vectorizer is None:
        return None

    cleaned = clean_text(text)
    matrix = count_vectorizer.transform([cleaned])

    if matrix.nnz == 0:
        return None

    probabilities = lda_model.transform(matrix)[0]
    dominant_index = probabilities.argmax()

    return probabilities, dominant_index


# --------------------------------------------------
# 9. WEBSITE HEADER
# --------------------------------------------------

st.title("Marathi-English NLP Analysis Dashboard")

st.write(
    "An intelligent text analysis platform for "
    "Marathi-English code-mixed language."
)

st.caption(
    "Technologies: NLP | TF-IDF | Linear SVM | "
    "Sentiment Analysis | Information Extraction | LDA"
)

st.divider()

# --------------------------------------------------
# 10. NAVIGATION
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "Text Analysis",
    "Topic Modelling",
    "About Project"
])


# ==================================================
# TAB 1: TEXT ANALYSIS
# ==================================================

with tab1:

    st.subheader("Analyze Marathi-English Text")

    user_input = st.text_area(
        "Enter your sentence:",
        placeholder="Example: Tu khup vait ahes",
        height=120
    )

    if st.button("Analyze Text", type="primary"):

        if not user_input.strip():
            st.warning("Please enter a sentence.")

        else:
            cleaned = clean_text(user_input)

            # SVM CLASSIFICATION
            vector = tfidf_vectorizer.transform([cleaned])
            prediction = model.predict(vector)[0]

            st.divider()
            st.subheader("1. Hate Speech Classification")

            if prediction == "HATE":
                st.error("HATE")

            elif prediction == "OFFENSIVE":
                st.warning("OFFENSIVE")

            else:
                st.success("NEUTRAL")

            st.write("Predicted Category:", prediction)

            # SENTIMENT ANALYSIS
            st.divider()
            st.subheader("2. Sentiment Analysis")

            sentiment, scores = analyze_sentiment(cleaned)

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Sentiment", sentiment)

            with col2:
                st.metric("Compound Score", scores["compound"])

            st.write("Positive:", scores["pos"])
            st.write("Negative:", scores["neg"])
            st.write("Neutral:", scores["neu"])

            st.caption(
                "VADER is primarily designed for English. "
                "Results on Romanized Marathi-English may be inaccurate."
            )

            # KEYWORD EXTRACTION
            st.divider()
            st.subheader("3. Information Extraction")

            keywords = extract_keywords(cleaned)

            st.write("Extracted Keywords:")

            if keywords:
                st.write(", ".join(
                    [word for word, count in keywords]
                ))
            else:
                st.write("No significant keywords identified.")

            # TARGET GROUP EXTRACTION
            st.write("Possible Target Groups:")

            groups = extract_target_groups(cleaned)

            for group in groups:
                st.info(group)

            st.caption(
                "Target-group extraction uses a configurable "
                "keyword dictionary and may miss implicit references."
            )

            # LDA FOR INDIVIDUAL SENTENCE
            st.divider()
            st.subheader("4. Topic Analysis")

            topic_result = predict_document_topic(cleaned)

            if topic_result is None:
                st.info(
                    "LDA model files are not available, or the "
                    "sentence contains no recognized vocabulary."
                )

            else:
                probabilities, dominant_index = topic_result

                st.write(
                    f"Dominant Topic: Topic {dominant_index + 1}"
                )

                st.write("Topic probabilities:")

                for i, probability in enumerate(probabilities):
                    st.write(
                        f"Topic {i + 1}: {probability:.2%}"
                    )

                    st.progress(float(probability))


# ==================================================
# TAB 2: TOPIC MODELLING
# ==================================================

with tab2:

    st.subheader("LDA Topic Explorer")

    if lda_model is None or count_vectorizer is None:

        st.warning(
            "LDA model files are not loaded yet. "
            "Save your trained LDA model and CountVectorizer "
            "to enable this dashboard."
        )

    else:

        topics = get_lda_topics()

        st.write(
            f"Total topics discovered: {len(topics)}"
        )

        for topic in topics:

            with st.expander(topic["topic"], expanded=True):
                st.write("Important words:")
                st.write(topic["words"])


# ==================================================
# TAB 3: ABOUT PROJECT
# ==================================================

with tab3:

    st.subheader("About the Project")

    st.write(
        """
        This project focuses on Marathi-English code-mixed
        hate speech detection using Natural Language Processing
        and Machine Learning.
        """
    )

    st.markdown("""
    **Main components:**

    - Text preprocessing and normalization
    - TF-IDF feature extraction
    - Linear SVM classification
    - Sentiment analysis
    - Keyword and target-group extraction
    - LDA topic modelling
    - Streamlit web application
    """)

    st.info(
        "The classification model predicts HATE, OFFENSIVE "
        "and NEUTRAL categories. Other analysis modules provide "
        "additional linguistic insights."
    )

st.divider()

st.caption(
    "Automated predictions may be incorrect, particularly "
    "for code-mixed language and context-dependent expressions."
)
