import os

import streamlit as st


st.title("Text Summarizer Comparisons")


PAGES_DIR = "pages"
EXTRACTIVE_DIR = os.path.join(PAGES_DIR, "extractive")
ABSTRACTIVE_DIR = os.path.join(PAGES_DIR, "abstractive")

general_pages = [st.Page("pages/general.py", title="General")]
extractive_pages = []
abstractive_pages = []


# TODO: Replace with def
for file in sorted(os.listdir(EXTRACTIVE_DIR)):
    file_path = os.path.join(EXTRACTIVE_DIR, file)
    if not os.path.isfile(file_path):
        continue

    page_name = file.rsplit(".", 1)[0].replace("_", "-").upper()
    extractive_pages.append(st.Page(file_path, title=page_name))

for file in sorted(os.listdir(ABSTRACTIVE_DIR)):
    file_path = os.path.join(ABSTRACTIVE_DIR, file)
    if not os.path.isfile(file_path):
        continue

    page_name = file.rsplit(".", 1)[0].replace("_", "-").upper()
    abstractive_pages.append(st.Page(file_path, title=page_name))

st.set_page_config(page_title="Summarization App", page_icon="📚", layout="wide")

navigation_groups = {
    "": general_pages,
    "✂️ Extractive approaches": extractive_pages,
    "✍️ Abstractive approaches": abstractive_pages,
}


if st.sidebar:
    pg = st.navigation(navigation_groups)
    pg.run()
