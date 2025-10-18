import streamlit as st

from utils.summarizers import ExtractiveAllowedModels, ExtractiveSummarizerModel

st.markdown(
    """
    This page describes **ALBERT (A Lite BERT)**, a model developed by Google Research. It is a variant of BERT (Bidirectional Encoder Representations from Transformers) optimized for **parameter efficiency**.

    Unlike abstractive models (like BART or T5) that *generate* new text, ALBERT is an **encoder-only** model. In summarization, its primary use is for *extractive* methods: it excels at reading text and identifying which existing sentences are the most important.

    ---

    ### Model Details

    * **Model:** `albert-base-v2`
    * **Type:** Extractive Summarization / Language Understanding
    * **Architecture:** Transformer Encoder

    ### Key Features: Parameter Reduction

    ALBERT's main innovation is its dramatic reduction in size compared to BERT, which it achieves through two techniques:

    1.  **Factorized Embedding:** It splits the large vocabulary embedding matrix (e.g., 30,000 words) into two smaller matrices. This saves millions of parameters.
    2.  **Cross-Layer Parameter Sharing:** All 12 layers of the Transformer encoder *share the same set of weights*. This is the biggest source of parameter reduction.

    **Result:** `albert-base-v2` has only **~12 million** parameters, compared to `bert-base-uncased` which has **~110 million**.

    ### Pre-training Objectives

    * **Masked Language Modeling (MLM):** The same as BERT; the model learns to predict randomly hidden ("masked") words in a sentence.
    * **Sentence-Order Prediction (SOP):** This *replaces* BERT's "Next Sentence Prediction." Instead of predicting if two sentences are consecutive, ALBERT must predict if two sentences are in their correct original order or if they have been swapped. This helps it learn about coherence and discourse.

    ### Strengths & Weaknesses

    * 👍 **Extremely Lightweight:** Has a very small memory footprint, making it easier to deploy on devices with limited resources.
    * 👍 **Good for Extractive Tasks:** Excellent at creating high-quality text representations. It can be fine-tuned to score sentences for an extractive summarizer (e.g., "how relevant is this sentence to the whole document?").
    * 👎 **Not for Generation:** As an encoder-only model, it **cannot** perform abstractive summarization (it can't *write* new sentences).
    * 👎 **Computationally Slow:** Despite having few parameters, it is *not* faster than BERT. It must still perform the full computation for all 12 layers (they just happen to share weights).

    [Link to Model Card on Hugging Face](https://huggingface.co/albert-base-v2)
    """
)

st.subheader("Enter text to summarize.")
text = st.text_area("Paste your text here:", height=250, placeholder="Type or paste text for summarization...")
submit = st.button("Summarize")

cols = st.columns(2)
if text and submit:
    model = ExtractiveSummarizerModel(model_name=ExtractiveAllowedModels.DISTILBERT_BASE, text=text)
    fig = model.plot_elbow()
    summarize = model.generate_summarize()
    st.subheader(ExtractiveAllowedModels.DISTILBERT_BASE.upper())
    st.markdown(f"<p style='text-align: justify'>{summarize}</p>", unsafe_allow_html=True)
    st.pyplot(fig)
