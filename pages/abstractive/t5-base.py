import streamlit as st
from utils.summarizers import AbstractiveAllowedModels, AbstractiveSummarizerModel


st.set_page_config(layout="wide")

st.markdown(
    """
    This page features **T5 (Text-to-Text Transfer Transformer)**, a highly flexible and powerful model from Google AI.

    The core idea of T5 is to reframe *every* NLP task (including summarization, translation, classification, and question answering) as a "text-to-text" problem. It takes text as input and generates new text as output.

    ---

    ### Model Details

    * **Model:** `t5-small`, `t5-base`, `t5-large` (and many fine-tuned variants)
    * **Type:** Abstractive Summarization (and many other tasks)
    * **Architecture:** Standard Transformer (Encoder-Decoder).

    ### The "Text-to-Text" Framework

    T5 uses specific prefixes to tell the model what task to perform. For summarization, the input text is simply prepended with a prefix like: **"summarize: "**

    * **Input:** `"summarize: [Article text]..."`
    * **Output:** `"[Generated summary text]..."`

    This unified framework allows T5 to learn a wide variety of tasks using the same model, architecture, and training objective.

    ### Pre-training Objective

    T5 is pre-trained on the massive **C4 (Colossal Clean Crawled Corpus)** dataset. Its objective is a variation of masked language modeling:

    1.  **Corruption:** Random spans (sequences) of tokens are masked in the input.
    2.  **Reconstruction:** The model is trained to generate *only* the masked-out spans, in the order they appeared.

    ### Strengths & Weaknesses

    * 👍 **Incredibly Versatile:** A single T5 model can be fine-tuned to perform many different tasks just by changing the input prefix.
    * 👍 **Strong Performance:** Achieves state-of-the-art or near-state-of-the-art results across a wide range of benchmarks, including summarization.
    * 👎 **Prefix Dependent:** The model's output is highly dependent on being given the correct prefix (e.g., `summarize:`).
    * 👎 **Large & Slow:** Like other large Transformers, T5 models (especially `base` and `large`) are computationally expensive and can be slow to run inference.

    [Link to Model Card on Hugging Face](https://huggingface.co/t5-base)
    """
)


st.subheader("Enter text to summarize.")
text = st.text_area("Paste your text here:", height=250, placeholder="Type or paste text for summarization...")
submit = st.button("Summarize")

cols = st.columns(2)
if text and submit:
    model = AbstractiveSummarizerModel(AbstractiveAllowedModels.T5_BASE, text)
    summarize = model.generate_summarize()
    st.subheader("T5-Base")
    st.markdown(f"<p style='text-align: justify'>{summarize}</p>", unsafe_allow_html=True)
