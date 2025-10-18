import streamlit as st

from utils.summarizers import ExtractiveAllowedModels, ExtractiveSummarizerModel

st.markdown(
    """
    This page describes **BERT (Bidirectional Encoder Representations from Transformers)**, the foundational model developed by Google that revolutionized modern NLP.

    BERT is an **encoder-only** model, meaning its primary job is to read text and build a deep, contextual understanding of it. It is the basis for most *extractive* summarization methods, where the model *scores* existing sentences for importance rather than writing new ones.

    ---

    ### Model Details

    * **Model:** `bert-base-uncased` (or `bert-base-cased`)
    * **Type:** Language Understanding / Extractive Summarization
    * **Architecture:** Transformer Encoder (12 layers, 768 hidden units, 12 attention heads)
    * **Parameters:** ~110 Million

    ### Bidirectional Context

    Before BERT, models read text either left-to-right (like GPT) or right-to-left. BERT's key innovation was to read the *entire sentence at once*, allowing it to understand context from both directions simultaneously.

    For example, in the sentence "He went to the **bank** to get cash," BERT can use the word "cash" (which appears later) to know that "**bank**" means a financial institution, not a river bank.

    ### Pre-training Objectives

    BERT was pre-trained on a massive dataset (Wikipedia & BookCorpus) using two main tasks:

    1.  **Masked Language Model (MLM):** 15% of the words in a sentence were randomly hidden (`[MASK]`). The model's job was to predict the correct hidden word based on the surrounding context.
    2.  **Next Sentence Prediction (NSP):** The model was given two sentences (A and B) and had to predict if B was the *actual* next sentence that followed A in the original text, or just a random sentence. This taught it to understand sentence relationships and coherence.

    ### Strengths & Weaknesses

    * 👍 **Powerful Feature Extractor:** Unmatched at its release for creating rich numerical representations (embeddings) of text for tasks like classification or question answering.
    * 👍 **Robust Contextual Understanding:** Its bidirectional nature makes it extremely accurate for tasks that require deep context.
    * 👎 **Not for Generation:** As an encoder-only model, it **cannot** generate new text (it can't perform abstractive summarization).
    * 👎 **Slower than Newer Models:** It is computationally heavy compared to newer, optimized models like ALBERT or DistilBERT.

    [Link to Model Card on Hugging Face](https://huggingface.co/bert-base-uncased)
    """
)

st.subheader("Enter text to summarize.")
text = st.text_area("Paste your text here:", height=250, placeholder="Type or paste text for summarization...")
submit = st.button("Summarize")

cols = st.columns(2)
if text and submit:
    model = ExtractiveSummarizerModel(model_name=ExtractiveAllowedModels.BERT_BASE, text=text)
    fig = model.plot_elbow()
    summarize = model.generate_summarize()
    st.subheader(ExtractiveAllowedModels.BERT_BASE.upper())
    st.markdown(f"<p style='text-align: justify'>{summarize}</p>", unsafe_allow_html=True)
    st.pyplot(fig)
