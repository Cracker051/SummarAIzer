import string
from collections import defaultdict
from heapq import nlargest

import spacy
import streamlit as st
from spacy.tokens.span import Span

from utils.cache import RedisCache, wrap_cache


def text_to_token(text: Span) -> list[str]:
    from spacy.lang.en.stop_words import STOP_WORDS

    return [token.text for token in text if token.text not in string.punctuation and token.text not in STOP_WORDS]


def word_frequency(tokens: list[str]) -> dict[str, float]:
    word_frequencies = defaultdict(int)
    for word in tokens:
        word_frequencies[word] += 1

    # Normalize
    max_popular_count = max(word_frequencies.values())
    normalized_freq = {}

    for key, value in word_frequencies.items():
        normalized_freq[key] = value / max_popular_count

    return normalized_freq


@wrap_cache(cache=RedisCache)
def sentence_frequency(sentences: tuple[Span, ...], word_norm_freq: dict[str, float]) -> dict[str, float]:
    sent_freq = defaultdict(float)
    for sentence in sentences:
        for word in sentence:
            sent_freq[str(sentence)] += word_norm_freq.get(word.text, 0)
    return dict(sent_freq)


input_column, result_column = st.columns([1, 1.5], gap="medium")
nlp_pipeline = spacy.load("en_core_web_sm")  # English pipeline

st.markdown(
    """
    This page demonstrates one of the simplest methods for extractive summarization, based purely on **Word Frequency** (also known as Term Frequency or TF).

    This algorithm operates on a simple statistical assumption: **The most important sentences are those that contain the most frequent words.**

    ---

    ### How it Works

    This method does not involve any complex models; it's a direct count of words.

    1.  **Text Cleaning:** The text is lowercased, and punctuation is removed.
    2.  **Stop Word Removal:** Common, non-informative words (like 'it', 'is', 'the', 'and') are filtered out.
    3.  **Frequency Calculation:** The algorithm counts the occurrences of every remaining word in the *entire document* and stores these counts in a frequency table.
    4.  **Sentence Scoring:** Each sentence is then scored by summing up the frequency counts of every word it contains.
    5.  **Ranking:** Sentences are ranked by their score, from highest to lowest.
    6.  **Summary Creation:** The top N highest-scoring sentences are selected and presented in their original order to create the final summary.

    ### Strengths & Weaknesses

    * 👍 **Extremely Fast & Simple:** This is one of the fastest and easiest summarization algorithms to implement and understand.
    * 👍 **Provides a Baseline:** It serves as a good "common sense" baseline to measure more complex models against. If a complex neural network can't beat this simple method, it's not a very good model.
    * 👎 **No Context:** The algorithm has zero understanding of meaning. It doesn't know "car" and "vehicle" are related, and it treats "apple" (fruit) and "Apple" (company) as the same word.
    * 👎 **Lacks Keyword Rarity:** Unlike TF-IDF, this method does not distinguish between a word that is frequent *everywhere* and a word that is frequent in just *one* important sentence. It can be heavily skewed by common-but-not-stop-listed words.
    * 👎 **Poor Coherence:** The resulting summary is just a collection of high-scoring sentences and may not flow logically or make sense as a standalone text.
    """
)

st.subheader("Enter text to summarize.")
text = st.text_area("Paste your text here:", height=250, placeholder="Type or paste text for summarization...")
submit = st.button("Summarize")

cols = st.columns(2)
if text and submit:
    summary_text = nlp_pipeline(text.lower().strip())  # Pass text to summarize
    tokens = [token.text for token in summary_text]
    wf = word_frequency(tokens)
    sf = sentence_frequency(tuple(summary_text.sents), wf)  # Wrap for tuple to cache
    sentences_count = int(len(list(summary_text.sents)) * 0.3)  # Only 30% of all sentences
    summarization = " ".join(map(lambda x: x.capitalize(), nlargest(sentences_count, sf, key=sf.get)))
    st.markdown(f"<p style='text-align: justify'>{summarization}</p>", unsafe_allow_html=True)
