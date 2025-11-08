import streamlit as st

from utils.text import summarize_by_som

st.markdown(
    """
    This page describes the **Self-Organizing Map (SOM)** approach, also known as a **Kohonen Map**, a clustering model adapted for NLP.

    A SOM is an **unsupervised neural network** model. For summarization, its job is to read all sentence vectors and organize them onto a 2D grid based on semantic similarity. It is used for *extractive* summarization, where the model *groups* sentences into topic clusters and selects representatives from the largest clusters.

    ---

    ### Model Details

    * **Model:** Self-Organizing Map (SOM) + Word2Vec
    * **Type:** Unsupervised Clustering / Extractive Summarization
    * **Architecture:** A 2D grid of competitive neurons, combined with a sentence vectorization model (e.g., `Word2Vec` or `BERT`).
    * **Parameters:** Varies based on the grid size (e.g., 10x10) and the dimensionality of the input vectors (e.g., 100 from Word2Vec).

    ### Semantic Clustering

    Unlike models that learn linguistic rules, a SOM learns the "shape" of the data. It maps high-dimensional sentence vectors onto a simple 2D grid. Sentences with similar meanings are mapped to the same or nearby neurons on the grid.

    For example, in a text about finance, sentences like "The **stock** rose sharply" and "Market **shares** hit a new high" would be mapped close together. A sentence like "He bought **stock** for the soup" would be mapped to a distant neuron, as its sentence vector would be completely different.

    *(Note: The  tag is a placeholder. To display an actual image, use `st.image("path/to/your/image.png")`)*

    ### Training Process

    The SOM is trained *on the specific document* it needs to summarize. It does not require a large external corpus.

    1.  **Sentence Vectorization:** The text is split into sentences. A model (like `Word2Vec`) is trained on these sentences to create word vectors. Each sentence is then converted into a single sentence vector (e.g., by averaging its word vectors).
    2.  **Competitive Learning (SOM Training):** The model iterates through the sentence vectors. For each vector, it finds the "Best Matching Unit" (BMU) or "winner"—the neuron on the 2D grid whose weights are most similar. The weights of this winning neuron (and its neighbors) are then adjusted to become even more similar to the input sentence vector. This process pulls neurons toward the sentence vectors, effectively grouping similar sentences around the same neurons.

    ### Strengths & Weaknesses

    * 👍 **Fully Unsupervised:** Requires no labeled data and trains only on the input text, making it highly adaptable to specific documents.
    * 👍 **Visualizable:** The resulting 2D map provides a clear visualization of the main topics and how they relate to each other.
    * 👎 **Not for Generation:** As a clustering method, it is strictly extractive and **cannot** generate new text (it cannot perform abstractive summarization).
    * 👎 **No World Knowledge:** Its understanding is limited *only* to the context of the input document. It has no external knowledge (unlike BERT).
    * 👎 **Sensitive to Vectorization:** The summary quality is highly dependent on the quality of the initial sentence vectors (e.g., from Word2Vec). Poor vectors will lead to poor clusters.
    """
)

st.subheader("Enter text to summarize.")
text = st.text_area("Paste your text here:", height=250, placeholder="Type or paste text for summarization...")
submit = st.button("Summarize")

cols = st.columns(2)
if text and submit:
    summarize, fig = summarize_by_som(text, ratio=0.3)
    st.subheader("KOHONEN MAPS")
    st.markdown(f"<p style='text-align: justify'>{summarize}</p>", unsafe_allow_html=True)
    st.pyplot(fig)
