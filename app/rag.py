from pathlib import Path
from typing import List
from openai import OpenAI

from .vectorstore import VectorStore, Document


class RagPipeline:
    def __init__(self, vector_store: VectorStore, max_chunks: int = 3):
        self.vector_store = vector_store
        self.max_chunks = max_chunks

    @classmethod
    def from_local_docs(cls, data_dir: str, client: OpenAI, max_chunks: int = 3) -> "RagPipeline":
        data_path = Path(data_dir)
        docs: List[Document] = []

        for idx, file in enumerate(sorted(data_path.glob("*.md"))):
            text = file.read_text(encoding="utf-8")
            docs.append(
                Document(
                    id=f"doc-{idx}",
                    text=text.strip(),
                    metadata={"filename": file.name},
                )
            )

        vector_store = VectorStore(client=client)
        if docs:
            vector_store.add_documents(docs)

        return cls(vector_store=vector_store, max_chunks=max_chunks)

    def retrieve(self, query: str) -> List[Document]:
        return self.vector_store.similarity_search(query, k=self.max_chunks)

    def build_context(self, query: str):
        docs = self.retrieve(query)
        chunks = []

        for i, d in enumerate(docs):
            header = f"[Document {i+1} | {d.metadata.get('filename', d.id)}]"
            chunks.append(f"{header}\n{d.text}\n")

        context_text = "\n\n".join(chunks)
        return context_text, docs
