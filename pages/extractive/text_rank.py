import matplotlib
import streamlit as st

from utils import text as text_utils


matplotlib.use("Agg")

st.set_page_config(page_title="Text Summarizer", layout="wide")

st.markdown(
    """
    This page demonstrates **TextRank**, a graph-based algorithm for extractive summarization. It was inspired directly by Google's **PageRank** algorithm (which ranks websites).

    Instead of ranking websites, TextRank ranks *sentences*. The core idea is that a sentence is "important" if it is similar to many other "important" sentences.

    ---

    ### How it Works

    1.  **Sentence Segmentation:** The text is split into individual sentences.
    2.  **Graph Building:** A graph is created where each sentence is a node (a "vertex").
    3.  **Edge Creation:** An "edge" (a connection) is drawn between two sentence nodes if they are similar to each other. "Similarity" is typically measured by how many words (or concepts) they share.
    4.  **Ranking (PageRank):** The PageRank algorithm is run on this graph. Sentences that are highly connected to other highly-connected sentences will "vote" for each other, accumulating a high score. This "importance" score is passed from node to node until the algorithm stabilizes.
    5.  **Selection:** The sentences are ranked by their final PageRank score.
    6.  **Summary Creation:** The top N highest-scoring sentences are selected and put back in their original order to form the summary.

    ### Strengths & Weaknesses

    * 👍 **Unsupervised:** It requires no pre-trained model or training data. It works directly on the input text.
    * 👍 **Language Independent:** Like TF-IDF, it works for most languages as long as the text can be split into sentences and words.
    * 👍 **Better Coherence:** By measuring sentence similarity, TextRank tends to select sentences that are central to the main topic, which can result in a more focused and coherent summary than simple frequency methods.
    * 👎 **Computationally Heavier:** Building the similarity matrix and running the PageRank algorithm is slower and more resource-intensive than TF-IDF or simple frequency counting.
    * 👎 **No True Understanding:** It is still a statistical method. It does not "understand" the text. It just measures word overlap.
    * 👎 **Bias Toward "Central" Sentences:** It might miss important but "unique" sentences (like a concluding sentence) that don't share many words with the rest of the text.
    """
)

input_column, result_column = st.columns([1, 1.5], gap="medium")

with input_column:
    st.subheader("Enter Text")
    text_input = st.text_area(
        "Paste your text here:", height=300, placeholder="Type or paste text for summarization..."
    )

    top_n_nodes = st.slider("Number of keywords to display in graph:", min_value=5, max_value=50, value=25, step=5)
    submit = st.button("Summarize")


with result_column:
    st.subheader("Summary & Graph")
    if text_input and submit:
        with st.spinner("Processing text and building graph..."):
            words = text_utils.full_preprocess_pipeline(text_input)

            if not words:
                st.warning("No valid words found after preprocessing. Please enter more text.")
            else:
                G = text_utils.build_co_occurrence_graph(words, window_size=4)

                pagerank_scores = text_utils.apply_pagerank(G)

                if not pagerank_scores:
                    st.warning("Could not calculate keyword scores. The text might be too short.")
                else:
                    fig = text_utils.generate_graph(G, pagerank_scores, top_n=top_n_nodes)
                    st.pyplot(fig)
    else:
        st.info("Enter text on the left to see the summary and graph here.")

st.markdown("<hr style='border:1px solid #E5E7EB'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#9CA3AF;'>Developed by Dmytro Lavreniuk & Vasyl Melnyk</p>",
    unsafe_allow_html=True,
)
