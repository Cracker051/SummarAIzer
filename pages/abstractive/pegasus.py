import streamlit as st

from utils.summarizers import AbstractiveAllowedModels, AbstractiveSummarizerModel


st.markdown(
    """
    This page features **PEGASUS (Pre-training with Extracted Gap-sentences for Abstractive Summarization)**, a state-of-the-art model from Google designed specifically for abstractive summarization.

    PEGASUS is renowned for its ability to generate extremely high-quality, human-like summaries by using a unique pre-training objective that mimics the summarization task itself.

    ---

    ### Model Details

    * **Model:** `google/pegasus-cnn_dailymail` (or other variants like `pegasus-xsum`, `pegasus-large`)
    * **Type:** Abstractive Summarization
    * **Architecture:** Standard Transformer (Encoder-Decoder).

    ### The PEGASUS Pre-training Objective: GSG

    The key innovation in PEGASUS is its pre-training task, **Gap-Sentences Generation (GSG)**:

    1.  **Select Sentences:** During pre-training (on massive web text), several important sentences are removed from a document.
    2.  **Generate Sentences:** The model is trained to *generate* these "gap-sentences" from the remaining text.
    3.  **Result:** This task forces the model to learn how to understand the main ideas of a document and express them fluently, which is almost identical to the task of summarization.

    ### Strengths & Weaknesses

    * 👍 **State-of-the-Art Quality:** Often achieves the highest ROUGE scores (a metric for summary quality) on many benchmark datasets.
    * 👍 **Highly Abstractive:** Excellent at paraphrasing, re-framing, and synthesizing information from multiple parts of the text into new, concise sentences.
    * 👎 **Very Computationally Heavy:** PEGASUS (especially the `large` version) is one of the most demanding summarization models, requiring significant GPU resources and time.
    * 👎 **Domain-Specific:** The best-performing models are fine-tuned on specific datasets (e.g., `pegasus-cnn_dailymail` for news, `pegasus-xsum` for "extreme" one-sentence summaries). Its performance may be lower on text that is very different from its training data.

    [Link to Model Card on Hugging Face](https://huggingface.co/google/pegasus-large)
    """
)
text = st.text_area(
    "Paste your text here:", height=250, max_chars=500, placeholder="Type or paste text for summarization..."
)
submit = st.button("Summarize")

cols = st.columns(2)
if text and submit:
    model = AbstractiveSummarizerModel(AbstractiveAllowedModels.PEGASUS, text)
    summarize = model.generate_summarize()
    st.subheader("PEGASUS Abstractive")
    st.markdown(f"<p style='text-align: justify'>{summarize}</p>", unsafe_allow_html=True)
