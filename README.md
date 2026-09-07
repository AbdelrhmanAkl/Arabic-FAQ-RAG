# Arabic FAQ RAG

Production Arabic FAQ retrieval and intent-aware reranking system.

## Architecture

Arabic Query
→ multilingual-e5-base
→ FAISS
→ Top-K Retrieval
→ Intent Classification
→ Intent-Aware Reranking

## Core Components

- multilingual-e5-base
- FAISS IndexFlatIP
- TF-IDF + LinearSVC intent classifier
- Intent-aware reranking

## Note

The Streamlit application and final documentation will be added in the next steps.
