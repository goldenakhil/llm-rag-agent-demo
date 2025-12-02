import os
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass
class Document:
    id: str
    text: str
    metadata: Dict[str, Any]


class VectorStore:
    def __init__(self, client: OpenAI, embedding_model: str = EMBEDDING_MODEL):
        self.client = client
        self.embedding_model = embedding_model
        self.documents: List[Document] = []
        self.embeddings: np.ndarray | None = None

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype="float32")

    def add_documents(self, docs: List[Document]) -> None:
        texts = [d.text for d in docs]
        new_embs = self._embed_texts(texts)

        if self.embeddings is None:
            self.embeddings = new_embs
        else:
            self.embeddings = np.vstack([self.embeddings, new_embs])

        self.documents.extend(docs)

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        if not self.documents or self.embeddings is None:
            return []

        query_vec = self._embed_texts([query])[0]

        doc_embs = self.embeddings
        dot = doc_embs @ query_vec
        norm_docs = np.linalg.norm(doc_embs, axis=1) + 1e-8
        norm_q = np.linalg.norm(query_vec) + 1e-8
        sims = dot / (norm_docs * norm_q)

        top_indices = np.argsort(-sims)[:k]
        return [self.documents[i] for i in top_indices]
