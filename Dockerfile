FROM astral/uv:python3.12-bookworm-slim

RUN groupadd --system --gid 999 nonroot && \
    useradd --system --gid 999 --uid 999 --create-home nonroot

WORKDIR /summarizator

ENV UV_TOOL_BIN_DIR=/usr/local/bin

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --compile-bytecode --link-mode=copy --no-install-project --no-dev && \
    uv run python3 -m nltk.downloader punkt punkt_tab stopwords wordnet && \
    uv pip install en_core_web_sm && \
    uv run python3 -m spacy link en_core_web_sm en

COPY . /summarizator

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

ENV PATH="/summarizator/.venv/bin:$PATH"

EXPOSE 8502

HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health

ENTRYPOINT [ "streamlit", "run", "main.py", "--server.address", "0.0.0.0", "--server.port", "8502" ]