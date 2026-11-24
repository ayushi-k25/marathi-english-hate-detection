# Marathi-English Code-Mixed Hate Speech Detection

## Project Overview

Marathi-English Code-Mixed Hate Speech Detection is a Natural Language Processing and Machine Learning project developed to identify and classify Marathi-English sentences into three categories: HATE, OFFENSIVE and NEUTRAL.

The project focuses on understanding mixed-language text, which is commonly used in everyday conversations and social media. It uses a Linear Support Vector Machine (SVM) classifier with TF-IDF feature extraction to predict the category of an input sentence.

The application is built using Streamlit and deployed on Render so that users can access it through a web browser.

## Objectives

- Identify hateful and offensive content in Marathi-English text.
- Classify sentences into three categories: HATE, OFFENSIVE and NEUTRAL.
- Apply different NLP preprocessing techniques to prepare text for analysis.
- Perform sentiment analysis and extract useful information from text.
- Discover common themes using topic modelling.
- Experiment with Generative AI to produce simple explanations of classification results.
- Make the application accessible through cloud deployment.

## Dataset Description

The project uses a dataset of 10,000 Marathi-English code-mixed text records, primarily written in Roman script.

The dataset contains three classes:

| Category | Number of Records |
|---|---:|
| HATE | 2,500 |
| OFFENSIVE | 2,500 |
| NEUTRAL | 5,000 |
| **Total** | **10,000** |

### Dataset Split

The dataset was divided into three parts:

| Split | Percentage | Records |
|---|---:|---:|
| Training | 80% | 8,000 |
| Validation | 10% | 1,000 |
| Testing | 10% | 1,000 |

## Technologies Used

- Python
- Pandas and NumPy
- Natural Language Processing (NLP)
- Scikit-learn
- TF-IDF and Bag of Words
- Linear Support Vector Machine (SVM)
- Latent Dirichlet Allocation (LDA)
- Streamlit
- Hugging Face Transformers
- Qwen2.5-0.5B-Instruct
- GitHub
- Render

## NLP Workflow

The following steps were performed to prepare and analyse the dataset:

1. Dataset loading and auditing
2. Text extraction and cleaning
3. Lowercasing and text normalization
4. Tokenization
5. Stopword removal
6. Stemming
7. Bag of Words feature extraction
8. TF-IDF vectorization
9. Sentiment analysis
10. Target-group identification
11. Keyword extraction
12. Topic modelling using LDA

These steps help convert raw text into a format that can be analysed and used for machine learning.

## Machine Learning Model

A Linear Support Vector Machine (SVM) was used for three-class text classification.

TF-IDF was used to convert text into numerical features. Unigrams and bigrams were considered to capture individual words as well as two-word combinations.

The trained model predicts one of the following categories:

- HATE
- OFFENSIVE
- NEUTRAL

The model was evaluated using the prepared test dataset and standard classification metrics.

## Topic Modelling

Latent Dirichlet Allocation (LDA) was used to identify common themes and patterns within the text.

Five topics were generated, with each topic represented by a group of frequently occurring words. This helped explore the underlying themes present in the dataset.

## Generative AI Experiment

Qwen2.5-0.5B-Instruct was experimented with to generate simple explanations of machine learning classification results.

The SVM model performs the actual classification, while the language model generates an explanation based on the predicted category.

## Web Application and Cloud Deployment

A Streamlit application was developed to allow users to enter Marathi-English sentences and view their predicted categories.

The application was deployed on Render, making it accessible through a public web link.

GitHub was used to store the application code, trained model and supporting files.

**Live Application:** Add your Render application URL here.

## Project Structure

```text
Marathi-English-Hate-Detection/
│
├── app.py
├── requirements.txt
├── marathi_english_hate_speech_svm.pkl
├── marathi_english_tfidf_vectorizer.pkl
└── README.md
```

## How to Run the Project Locally

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**2. Open the project directory**

```bash
cd marathi-english-hate-detection
```

**3. Install the required libraries**

```bash
pip install -r requirements.txt
```

**4. Run the application**

```bash
streamlit run app.py
```

## Model Evaluation

The Linear SVM classifier achieved 100% accuracy on the prepared test dataset.

This result represents the model's performance on the available test split. Further evaluation on diverse, naturally occurring Marathi-English text is necessary to understand its real-world performance.

## Limitations

- Spelling variations in Marathi-English text can affect predictions.
- The model may struggle with sarcasm and context-dependent expressions.
- Unseen words may not be represented in the TF-IDF vocabulary.
- Incorrect predictions, including false positives and false negatives, are possible.
- The dataset can be expanded with more diverse and naturally written examples.

## Future Scope

- Expand the dataset with more real-world Marathi-English conversations.
- Improve the model's understanding of sentence context.
- Explore multilingual transformer-based models.
- Reduce incorrect classifications.
- Extend the system to support additional regional languages.
- Improve the explanations generated using Generative AI.

## Project Information

**Project Title:** Marathi-English Code-Mixed Hate Speech Detection

**Domain:** Natural Language Processing and Machine Learning

**Classification Algorithm:** Linear Support Vector Machine

**Feature Extraction:** TF-IDF

**Web Framework:** Streamlit

**Cloud Platform:** Render

---
# Deploy link:
https://marathi-english-hate-detection.onrender.com

Developed as an academic project to explore text classification, multilingual language processing and machine learning-based content detection.
