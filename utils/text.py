import re
import string
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import nltk
from matplotlib.figure import Figure
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


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
