# Summariztor

This repository contains the implementation of a project focused on the comparative analysis of automatic text summarization methods. The system evaluates extractive and abstractive approaches using 12 distinct algorithms within a unified web interface.

## Core Features

* **Extractive Methods:** Implementation of TextRank, TF-IDF, and Transformer-based encoders (BERT, ALBERT, XLNet, XLM-RoBERTa).
* **Abstractive Methods:** Implementation of generative architectures (BART, T5, PEGASUS).
* **Clustering:** Utilization of Kohonen Self-Organizing Maps (SOM) combined with Word2Vec embeddings.
* **Optimized Performance:** Redis-based caching layer for model inference results.

## Prerequisites

* Docker Compose

## Installation and Configuration

### 1. Environment Setup

Configure the application by creating a .env file from the provided template:

```bash
cp dist.env .env
```

Open the .env file and define required variables

### 2. Deployment

Execute the following commands to build and launch the application:

```bash
docker compose build
docker compose up
```

The interface will be accessible at: http://localhost:8501

## System Architecture

The application is structured as a multi-container Docker environment:

1. **Application Container:** Runs the Streamlit frontend and Python NLP engine.
2. **Cache Container:** Redis instance used to store and retrieve previously computed summaries to minimize resource consumption.

## Tech Stack

* **Language:** Python >=3.12.11
* **Frontend Framework:** Streamlit
* **ML Libraries:** Hugging Face Transformers, NLTK, spaCy, PyTorch, MiniSom
* **Infrastructure:** Docker, Redis