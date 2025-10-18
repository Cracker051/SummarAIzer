import streamlit as st

from utils.summarizers import ExtractiveAllowedModels, ExtractiveSummarizerModel


st.markdown(
    """
    This page describes **XLNet**, a model developed by Google AI and Carnegie Mellon University. It was designed to improve upon BERT by taking the best parts of both autoregressive (AR) models like GPT and autoencoding (AE) models like BERT.

    XLNet is an **encoder** model, making it suitable for *extractive* summarization tasks where it excels at understanding deep context and dependencies.

    ---

    ### Model Details

    * **Model:** `xlnet-base-cased`
    * **Type:** Language Understanding / Extractive Summarization
    * **Architecture:** Transformer (incorporating ideas from Transformer-XL)
    * **Parameters:** ~110 Million (similar to `bert-base`)

    ### Key Feature: Permutation Language Modeling (PLM)

    XLNet's core innovation is **Permutation Language Modeling (PLM)**. Instead of corrupting the input with `[MASK]` tokens like BERT, XLNet learns by predicting words in a random (permuted) order.

    * **BERT (Autoencoding):** Sees a *corrupted* sentence and predicts `[MASK]` tokens.
    * **GPT (Autoregressive):** Predicts the *next* word from left-to-right.
    * **XLNet (Permutation):** Predicts the *next* word based on a random ordering of all other words. For example, to predict `word_3` in a 5-word sentence, it might get `word_1`, `word_5`, and `word_2` as context.

    This method allows it to learn **bidirectional context** (like BERT) without suffering from the `[MASK]` token discrepancy (since `[MASK]` tokens don't appear in real-world fine-tuning).

    ### Other Innovations

    * **Two-Stream Self-Attention:** A complex attention mechanism that makes PLM possible, allowing the model to know *which* position to predict without seeing the *content* of the token at that position.
    * **Transformer-XL Integration:** It incorporates relative positional embeddings from Transformer-XL, making it exceptionally good at tasks that require understanding **long-range dependencies** in text.

    ### Strengths & Weaknesses

    * 👍 **Superior Performance:** When released, XLNet outperformed BERT on many major NLP benchmarks, including SQuAD (question answering) and GLUE (general language understanding).
    * 👍 **Excellent on Long Texts:** Its use of Transformer-XL architecture makes it one of the best models for tasks involving long documents where context from far away is important.
    * 👎 **Very Complex:** The two-stream attention and permutation-based training are significantly more complex and computationally expensive than BERT's.
    * 👎 **Not for Generation:** It is an encoder, not a decoder. It is designed to understand text, not to generate new, abstractive summaries.

    [Link to Model Card on Hugging Face](https://huggingface.co/xlnet-base-cased)
    """
)
st.subheader("Enter text to summarize.")
text = st.text_area("Paste your text here:", height=250, placeholder="Type or paste text for summarization...")
submit = st.button("Summarize")

cols = st.columns(2)
if text and submit:
    model = ExtractiveSummarizerModel(model_name=ExtractiveAllowedModels.XLNET_BASE, text=text)
    fig = model.plot_elbow()
    summarize = model.generate_summarize()
    st.subheader(ExtractiveAllowedModels.XLNET_BASE.upper())
    st.markdown(f"<p style='text-align: justify'>{summarize}</p>", unsafe_allow_html=True)
    st.pyplot(fig)
