import streamlit as st

from utils.summarizers import ExtractiveAllowedModels, ExtractiveSummarizerModel


st.markdown(
    """
    This page describes **`xlm-mlm-enfr-1024`**, a specific, bilingual version of the **XLM (Cross-lingual Language Model)** family developed by Facebook AI.

    Unlike the large 100-language models, this version was trained *specifically* on a corpus of **English and French** text. Its name breaks down as:
    * **XLM:** Cross-lingual Language Model
    * **MLM:** Pre-trained using the **Masked Language Modeling** objective (like BERT).
    * **enfr:** Trained on English and French.
    * **1024:** Uses a hidden unit size of 1024.

    ---

    ### Model Details

    * **Model:** `FacebookAI/xlm-mlm-enfr-1024`
    * **Type:** Bilingual Language Understanding
    * **Architecture:** Transformer Encoder (1024 hidden units, 8 attention heads)

    ### How it Works

    This model is an **encoder** designed to find a shared "understanding" of English and French. It was pre-trained by reading massive amounts of text in both languages and performing the **Masked Language Model (MLM)** task:

    1.  A sentence in either English or French is taken.
    2.  Words are randomly hidden (`[MASK]`).
    3.  The model's only goal is to predict the correct hidden words based on the surrounding context.

    By performing this task on both languages simultaneously with a shared vocabulary, the model is forced to create internal representations (embeddings) that are aligned. For example, it learns that the English word "**cat**" and the French word "**chat**" occupy a very similar "meaning space."

    ### Strengths & Weaknesses

    * 👍 **High-Quality Bilingual Embeddings:** Because it focuses *only* on English and French, its representations for these two languages are often more nuanced and accurate than a 100-language model (like `xlm-mlm-100-1280`) where the model's "attention" is split.
    * 👍 **Good for Transfer Learning:** It is excellent for "zero-shot" or "few-shot" transfer. You can fine-tune it on a task (like text classification) using only English data, and it will be able to perform the same task on French data with reasonable accuracy.
    * 👎 **Not for Generation:** As an encoder-only model, it cannot perform abstractive summarization or text generation.
    * 👎 **Bilingual Only:** It has no knowledge of any language other than English and French (unlike its multilingual counterparts XLM-R or `xlm-mlm-100-1280`).

    [Link to Model Card on Hugging Face](https://huggingface.co/FacebookAI/xlm-mlm-enfr-1024)
    """
)

st.subheader("Enter text to summarize.")
text = st.text_area("Paste your text here:", height=250, placeholder="Type or paste text for summarization...")
submit = st.button("Summarize")

cols = st.columns(2)
if text and submit:
    model = ExtractiveSummarizerModel(model_name=ExtractiveAllowedModels.XLM_MLM, text=text)
    fig = model.plot_elbow()
    summarize = model.generate_summarize()
    st.subheader(ExtractiveAllowedModels.XLM_MLM.upper())
    st.markdown(f"<p style='text-align: justify'>{summarize}</p>", unsafe_allow_html=True)
    st.pyplot(fig)
