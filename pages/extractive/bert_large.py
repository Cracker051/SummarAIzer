import streamlit as st

from utils.summarizers import ExtractiveAllowedModels, ExtractiveSummarizerModel

st.markdown(
    """
    This page describes **BERT-large**, the larger and more powerful counterpart to `bert-base`. It was released by Google as the high-performance option for researchers and teams with significant computational resources.

    Like its smaller sibling, BERT-large is an **encoder-only** model. It does not *generate* new text but is instead used for high-accuracy *extractive* tasks by deeply understanding the text it reads.

    ---

    ### Model Details

    * **Model:** `bert-large-uncased` (or `bert-large-cased`)
    * **Type:** Language Understanding / Extractive Summarization
    * **Architecture:** Deeper and wider Transformer Encoder
        * **Layers:** 24 (vs. 12 in `base`)
        * **Hidden Units:** 1024 (vs. 768 in `base`)
        * **Attention Heads:** 16 (vs. 12 in `base`)
    * **Parameters:** **~340 Million** (vs. ~110M in `base`)

    ### Pre-training Objectives

    Its training is identical to `bert-base`, using two unsupervised tasks on a massive corpus (Wikipedia & BookCorpus):

    1.  **Masked Language Model (MLM):** The model predicts randomly hidden (`[MASK]`) words in a sentence, forcing it to learn bidirectional context.
    2.  **Next Sentence Prediction (NSP):** The model predicts whether two given sentences were originally consecutive in the text, teaching it to understand coherence.

    ### Strengths & Weaknesses

    * 👍 **Higher Performance:** The primary reason for its existence. Its increased size allows it to capture more complex patterns, leading to state-of-the-art results (at its time of release) on benchmarks like GLUE, SQuAD, and others.
    * 👍 **Better Nuance:** More effective at understanding complex language, ambiguity, and subtle contextual clues.
    * 👎 **Very High Cost:** Requires significantly more GPU memory and processing power than the `base` model. It can be slow and expensive to fine-tune and run inference.
    * 👎 **Not for Generation:** It is an encoder, not a decoder. It cannot be used for abstractive summarization out-of-the-box.

    [Link to Model Card on Hugging Face](https://huggingface.co/bert-large-uncased)
    """
)

st.subheader("Enter text to summarize.")
text = st.text_area("Paste your text here:", height=250, placeholder="Type or paste text for summarization...")
submit = st.button("Summarize")

cols = st.columns(2)
if text and submit:
    model = ExtractiveSummarizerModel(model_name=ExtractiveAllowedModels.BERT_LARGE, text=text)
    fig = model.plot_elbow()
    summarize = model.generate_summarize()
    st.subheader(ExtractiveAllowedModels.BERT_LARGE.upper())
    st.markdown(f"<p style='text-align: justify'>{summarize}</p>", unsafe_allow_html=True)
    st.pyplot(fig)
