import re
import string
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
from gensim.models import Word2Vec
from matplotlib.figure import Figure
from minisom import MiniSom
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


def setup_nltk() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")


setup_nltk()


def preprocess_text(text: str) -> list[str]:
    words = text.lower().split()
    return words


def full_preprocess_pipeline(text: str) -> list[str]:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)

    tokens = word_tokenize(text)

    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return lemmatized_words


def build_co_occurrence_graph(words: list[str], window_size: int = 4) -> nx.Graph:
    G = nx.Graph()
    co_occurrences = defaultdict(int)

    for i in range(len(words) - window_size + 1):
        window = words[i : i + window_size]
        unique_words_in_window = list(set(window))

        if len(unique_words_in_window) < 2:
            continue

        for j in range(len(unique_words_in_window)):
            for k in range(j + 1, len(unique_words_in_window)):
                w1 = unique_words_in_window[j]
                w2 = unique_words_in_window[k]

                pair = tuple(sorted((w1, w2)))

                co_occurrences[pair] += 1

    for pair, weight in co_occurrences.items():
        G.add_edge(pair[0], pair[1], weight=weight)

    return G


def apply_pagerank(G: nx.Graph) -> dict:
    if not G:
        return {}
    try:
        pagerank_scores = nx.pagerank(G, weight="weight")
        return pagerank_scores
    except nx.PowerIterationFailedConvergence:
        # Повертає рівномірний розподіл, якщо PageRank не зійшовся
        return {node: 1.0 / len(G) for node in G.nodes()}


def generate_graph(G: nx.Graph, pagerank_scores: dict, top_n: int = 25) -> Figure:
    fig = plt.figure(figsize=(12, 8))

    sorted_scores = sorted(pagerank_scores.items(), key=lambda item: item[1], reverse=True)
    top_nodes = [node for node, score in sorted_scores[:top_n]]
    sub_G = G.subgraph(top_nodes)

    sub_pagerank_scores = {node: pagerank_scores[node] for node in top_nodes}

    if not sub_G:
        plt.title("No graph to display.")
        return fig

    pos = nx.spring_layout(sub_G)

    nx.draw_networkx_nodes(
        sub_G, pos, node_size=[v * 20000 for v in sub_pagerank_scores.values()], node_color="skyblue", alpha=0.8
    )

    nx.draw_networkx_edges(sub_G, pos, width=1.0, alpha=0.3, edge_color="gray")

    nx.draw_networkx_labels(sub_G, pos, font_size=10)

    plt.title(f"Top {len(sub_G.nodes())} Keywords Co-occurrence Graph")
    plt.axis("off")

    return fig


def sentence_vector(sentence_tokens, model) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    vectors = [model.wv[word] for word in sentence_tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def summarize_by_som(text: str, ratio: float = 0.3) -> tuple[str, plt.Figure]:
    # Split text into sentences
    sentences = sent_tokenize(text)

    # Tokenize each sentence into words
    tokenized_sentences = [word_tokenize(s.lower()) for s in sentences]

    w2v_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=2, epochs=50)

    sentence_vectors = np.array([sentence_vector(tokens, w2v_model) for tokens in tokenized_sentences])

    # Build Kohonen map
    som_size = int(np.ceil(np.sqrt(len(sentence_vectors))))  # grid size
    som = MiniSom(som_size, som_size, sentence_vectors.shape[1], sigma=0.5, learning_rate=0.5)
    som.random_weights_init(sentence_vectors)
    som.train_random(sentence_vectors, 100)

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, vec in enumerate(sentence_vectors):
        w = som.winner(vec)
        ax.text(
            w[0] + 0.5,
            w[1] + 0.5,
            str(i) + " " + " ".join(sentences[i].split()[:2]),
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(facecolor="skyblue", alpha=0.6, lw=0),
        )
    ax.set_title("Sentence Clusters (Kohonen Map)")
    ax.set_xlim([0, som_size])
    ax.set_ylim([0, som_size])
    ax.invert_yaxis()
    ax.grid(True)
    fig.suptitle("SOM Clustering Results", fontsize=16)

    # Get each sentence’s winning neuron
    winners = [som.winner(vec) for vec in sentence_vectors]

    # Cluster sentences by neuron position
    clusters = {}
    for i, w in enumerate(winners):
        clusters.setdefault(w, []).append(i)

    total_sentences = len(sentences)
    n = int(np.ceil(total_sentences * ratio))
    n = max(1, min(n, total_sentences))

    # Sort clusters by how many sentences they contain
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    selected_indices = [v[0] for k, v in sorted_clusters[:n]]

    summary = " ".join([sentences[i] for i in sorted(selected_indices)])
    return summary, fig
