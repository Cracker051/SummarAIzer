import re

import nltk
import numpy as np
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.cache import RedisCache, wrap_cache


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def preprocess_sentence(sentence) -> str:
    global lemmatizer
    sentence = sentence.lower()
    sentence = re.sub(r"[^a-z0-9\s]", "", sentence)
    tokens = sentence.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)


@wrap_cache(RedisCache)
def tfidf_summarize_advanced(text) -> str:
    sentences = nltk.sent_tokenize(text)

    processed_sentences = [preprocess_sentence(s) for s in sentences]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)

    mean_scores = np.array(tfidf_matrix.mean(axis=1)).ravel()
    max_scores = np.array(tfidf_matrix.max(axis=1).toarray()).ravel()

    lengths = np.array([len(s.split()) for s in sentences])
    length_weight = np.log1p(lengths)

    final_scores = (mean_scores + max_scores) * length_weight

    sentence_count = int(len(sentences) * 0.3)
    top_indices = final_scores.argsort()[-sentence_count:][::-1]

    summary = [sentences[i] for i in sorted(top_indices)]
    return " ".join(summary)


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

st.markdown(
    """
    This page demonstrates summarization using **TF-IDF**, a classic statistical method from Information Retrieval. It is *not* a neural network but a mathematical algorithm to score the importance of words.

    This is a purely **extractive** method. It identifies the most "important" sentences from the original text and combines them to form a summary.

    ---

    ### How it Works

    TF-IDF works by scoring every word in the document to determine how important it is. This score is based on two factors:

    1.  **Term Frequency (TF):** How often does a word appear in a *single sentence*? (A word that appears multiple times in one sentence is probably important *for that sentence*.)
    2.  **Inverse Document Frequency (IDF):** How rare is the word across *all sentences* (the entire document)? (Common words like "the" or "and" appear everywhere and get a low score, while unique keywords get a high score.)

    The final **TF-IDF score** for a word is `TF * IDF`. A high score means the word is frequent within its sentence but rare overall, making it a strong keyword.

    ### Summarization Steps

    1.  **Pre-processing:** The text is cleaned, and common "stop-words" (like 'a', 'the', 'is') are removed.
    2.  **Segmentation:** The entire text is split into individual sentences.
    3.  **TF-IDF Calculation:** The TF-IDF score is calculated for every remaining word in the document.
    4.  **Sentence Scoring:** Each sentence is given an "importance score" by summing the TF-IDF scores of all the words it contains.
    5.  **Ranking:** Sentences are ranked from highest score to lowest.
    6.  **Summary Creation:** The top N (e.g., top 3) highest-scoring sentences are selected and joined together, in their original order, to form the summary.

    ### Strengths & Weaknesses

    * 👍 **Very Fast & Simple:** Computationally cheap and easy to implement.
    * 👍 **Language Independent:** Works on any language as long as you can tokenize sentences and (ideally) have a stop-word list.
    * 👎 **No Context:** Does not understand meaning, semantics, or synonyms. "Car" and "auto" are treated as two completely different words.
    * 👎 **Bias Towards Long Sentences:** Longer sentences often get higher scores just because they contain more words, even if they aren't more important.
    * 👎 **Can Lack Coherence:** Simply picking the top 3 sentences doesn't guarantee they will flow together logically as a summary.
    """
)

st.subheader("Enter text to summarize.")
text = st.text_area("Paste your text here:", height=250, placeholder="Type or paste text for summarization...")
submit = st.button("Summarize")

cols = st.columns(2)
if text and submit:
    summarize = tfidf_summarize_advanced(text)
    st.subheader("TFIDF")
    st.markdown(f"<p style='text-align: justify'>{summarize}</p>", unsafe_allow_html=True)
