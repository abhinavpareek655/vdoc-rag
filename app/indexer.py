# app/indexer.py
import chromadb
import json
import os
import numpy as np


class ChromaIndexer:
    def __init__(self, embedding_function=None, persist_directory="./storage/chroma_db"):
        """
        Persistent Chroma DB (DuckDB + Parquet) wrapper.
        Stores vectors and metadata to disk so index survives restarts.
        """
        os.makedirs(persist_directory, exist_ok=True)
        self.embedding_function = embedding_function

        # Use the PersistentClient backed by the provided directory
        try:
            self.client = chromadb.PersistentClient(path=persist_directory)
        except Exception:
            # fallback to in-memory client if PersistentClient not available
            print("[indexer] PersistentClient not available, falling back to in-memory client.")
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            "vdoc",
            metadata={"description": "VDoc-RAG persistent storage"},
        )

        print(f"✅ Chroma index loaded from: {persist_directory}")

    def _sanitize_metadata(self, metadata):
        clean_meta = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                clean_meta[k] = v
            else:
                try:
                    clean_meta[k] = json.dumps(v)
                except Exception:
                    clean_meta[k] = str(v)
        return clean_meta

    def upsert(self, items):
        ids = [it[0] for it in items]
        embeddings = [it[1] for it in items]
        metadatas = [self._sanitize_metadata(it[2]) for it in items]
        documents = [it[3] for it in items]

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )
        print(f"💾 Upserted {len(items)} chunks into persistent Chroma collection.")

    def query(self, qvec, top_k=5):
        """
        qvec: numpy vector or list (query embedding)
        Returns list of {id, text, metadata, score (cosine sim 0–1)}
        """
        res = self.collection.query(
            query_embeddings=[qvec],
            n_results=top_k,
            include=["embeddings", "metadatas", "documents", "distances"],
        )

        out = []
        if not res or "ids" not in res or len(res["ids"]) == 0:
            return out

        qvec = np.array(qvec, dtype=np.float32)

        for i in range(len(res["ids"][0])):
            try:
                chunk_vec = np.array(res["embeddings"][0][i], dtype=np.float32)
                cos_sim = float(
                    np.dot(qvec, chunk_vec) / (np.linalg.norm(qvec) * np.linalg.norm(chunk_vec) + 1e-8)
                )
                cos_sim = max(0.0, min(1.0, cos_sim))
            except Exception:
                try:
                    dist = res.get("distances", [[0]])[0][i]
                    cos_sim = max(0.0, min(1.0, 1.0 - float(dist)))
                except Exception:
                    cos_sim = 0.0

            out.append({
                "id": res["ids"][0][i],
                "text": res.get("documents", [[None]])[0][i],
                "metadata": res.get("metadatas", [[None]])[0][i],
                "score": round(cos_sim, 4),
            })

        out.sort(key=lambda x: x["score"], reverse=True)
        return out
