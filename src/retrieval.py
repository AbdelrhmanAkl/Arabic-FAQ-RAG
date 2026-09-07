import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class ArabicFAQRetriever:
    """
    Production Arabic FAQ semantic retriever.

    Pipeline:
        User Query
            ↓
        multilingual-e5-base
            ↓
        FAISS IndexFlatIP
            ↓
        Top-K FAQ chunks
    """

    def __init__(
        self,
        faiss_path: str,
        metadata_path: str,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str = "cpu",
    ):
        self.faiss_path = faiss_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        self.device = device

        # Load FAISS index
        self.index = faiss.read_index(self.faiss_path)

        # Load metadata
        self.metadata = pd.read_parquet(self.metadata_path)

        # Validate FAISS ↔ metadata alignment
        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "FAISS index size does not match metadata row count."
            )

        # Load embedding model
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
        )

        # Validate embedding dimension
        if self.index.d != self.model.get_sentence_embedding_dimension():
            raise ValueError(
                "FAISS dimension does not match embedding model dimension."
            )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ):
        """
        Retrieve top-k FAQ chunks for an Arabic query.

        Returns:
            list[dict]
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty.")

        top_k = min(
            int(top_k),
            self.index.ntotal,
        )

        # E5 query prefix
        query_text = f"query: {query}"

        # Encode query
        query_embedding = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        query_embedding = query_embedding.astype("float32")

        # FAISS search
        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for rank, (score, corpus_idx) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            if corpus_idx < 0:
                continue

            row = self.metadata.iloc[int(corpus_idx)]

            results.append(
                {
                    "rank": rank,
                    "corpus_idx": int(corpus_idx),
                    "score": float(score),
                    "id": row["id"],
                    "company_id": row["company_id"],
                    "company": row["company"],
                    "domain": row["domain"],
                    "country": row["country"],
                    "chunk_idx": int(row["chunk_idx"]),
                    "chunk_text": row["chunk_text"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "answerable": bool(row["answerable"]),
                    "question_type": row["question_type"],
                }
            )

        return results