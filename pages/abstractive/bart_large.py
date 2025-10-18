import streamlit as st

from utils.summarizers import AbstractiveSummarizerModel, AbstractiveAllowedModels


st.markdown(
    """
    This page uses **BART (Bidirectional and Auto-Regressive Transformers)**, a powerful abstractive summarization model developed by Facebook AI. 
    
    Unlike extractive methods that just copy sentences, BART *rewrites* the text, generating new sentences that capture the core meaning, much like a human would.

    ---

    ### Model Details

    * **Model:** `facebook/bart-large-cnn`
    * **Type:** Abstractive Summarization
    * **Architecture:** BART is a sequence-to-sequence model with a bidirectional encoder (like BERT) and an auto-regressive decoder (like GPT). It's pre-trained by corrupting text (e.g., masking tokens, deleting sentences) and learning to reconstruct the original.
    * **Training Data:** This specific version was fine-tuned on the **CNN / Daily Mail** dataset, which consists of news articles and their corresponding human-written summaries (highlights).

    ### How it Works

    1.  **Encoding:** The full input text is fed into the bidirectional encoder, allowing the model to understand the context of every word, looking both forwards and backwards.
    2.  **Decoding:** The auto-regressive decoder then generates the summary token by token, attending to the encoded input to decide what information is most important to include and how to phrase it.

    ### Strengths & Weaknesses

    * 👍 **High Fluency:** Generates summaries that are highly readable, coherent, and grammatically correct.
    * 👍 **True Abstraction:** Can paraphrase and condense complex ideas into novel sentences.
    * 👎 **Slower:** More computationally intensive (requires more processing power and time) than simpler models.
    * 👎 **Risk of Hallucination:** Because it *generates* text, it can sometimes introduce information that wasn't in the original text (though it's generally good at staying factual).

    [Link to Model Card on Hugging Face](https://huggingface.co/facebook/bart-large-cnn)
    """
)

st.subheader("Enter text to summarize.")
text = st.text_area("Paste your text here:", height=250, placeholder="Type or paste text for summarization...")
submit = st.button("Summarize")

cols = st.columns(2)
if text and submit:
    model = AbstractiveSummarizerModel(AbstractiveAllowedModels.BART, text)
    summarize = model.generate_summarize()
    st.subheader("BART Abstractive")
    st.markdown(f"<p style='text-align: justify'>{summarize}</p>", unsafe_allow_html=True)
