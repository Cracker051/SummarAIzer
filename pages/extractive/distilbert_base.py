import streamlit as st

from utils.summarizers import ExtractiveAllowedModels, ExtractiveSummarizerModel


st.markdown(
    """
    This page describes **DistilBERT**, a smaller, faster, and lighter version of BERT developed by Hugging Face.

    It was created using a technique called **Knowledge Distillation**, where a smaller "student" model (DistilBERT) is trained to mimic the behavior of a larger "teacher" model (`bert-base`). The goal is to create a model that is much more efficient while retaining most of the original's performance.

    ---

    ### Model Details

    * **Model:** `distilbert-base-uncased`
    * **Type:** Language Understanding / Extractive Summarization
    * **Architecture:** A compressed Transformer Encoder
        * **Layers:** 6 (vs. 12 in `bert-base`)
        * **Parameters:** **~66 Million** (vs. ~110M in `bert-base`)

    ### How it Works: Knowledge Distillation

    DistilBERT was *not* trained on the Next Sentence Prediction task. Instead, it was trained on the same large corpus as BERT, using a special "triple loss" to learn from its teacher:

    1.  **Distillation Loss:** The student model learns to match the rich probability distributions of the teacher's output (its "soft predictions").
    2.  **Masked Language Model (MLM) Loss:** The standard BERT task of predicting hidden (`[MASK]`) words.
    3.  **Cosine Embedding Loss:** Pushes the student's hidden state vectors to be in the same direction as the teacher's vectors.

    ### Strengths & Weaknesses

    * 👍 **Fast & Lightweight:** It is **~40% smaller** than `bert-base` and runs **~60% faster**, making it ideal for production environments, real-time APIs, and on-device applications (like mobile phones).
    * 👍 **High Performance:** It retains **~97%** of BERT's language understanding capabilities (as measured on the GLUE benchmark), offering an excellent trade-off between speed and accuracy.
    * 👎 **Slightly Less Accurate:** As a compressed model, it is slightly less powerful than the full `bert-base` model and may not perform as well on highly complex or nuanced language tasks.
    * 👎 **Not for Generation:** Like all BERT-style models, it is an encoder-only model and cannot generate new text for abstractive summarization.

    [Link to Model Card on Hugging Face](https://huggingface.co/distilbert-base-uncased)
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
