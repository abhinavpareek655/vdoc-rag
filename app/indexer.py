# app/indexer.py
import chromadb
import json

class ChromaIndexer:
    def __init__(self, embedding_function=None, persist_directory="./chroma_db"):
        """
        Simple wrapper for new Chroma client (v0.5+)
        """
        self.embedding_function = embedding_function
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("vdoc")

    def _sanitize_metadata(self, metadata):
        """
        Convert unsupported types (dicts, lists, tuples) into JSON strings.
        """
        clean_meta = {}
        for k, v in metadata.items():
            # Allow simple types directly
            if isinstance(v, (str, int, float, bool)) or v is None:
                clean_meta[k] = v
            else:
                try:
                    clean_meta[k] = json.dumps(v)
                except Exception:
                    clean_meta[k] = str(v)
        return clean_meta

    def upsert(self, items):
        """
        items: list of (id, vector, metadata, text)
        """
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

    def query(self, qvec, top_k=5):
        """
        qvec: query embedding (np.ndarray or list)
        """
        res = self.collection.query(query_embeddings=[qvec], n_results=top_k)
        out = []
        if res and "ids" in res and len(res["ids"]) > 0:
            for i in range(len(res["ids"][0])):
                out.append({
                    "id": res["ids"][0][i],
                    "score": res["distances"][0][i],
                    "metadata": res["metadatas"][0][i],
                    "text": res["documents"][0][i],
                })
        return out
