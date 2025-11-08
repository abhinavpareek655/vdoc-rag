from sentence_transformers import SentenceTransformer
import numpy as np

class TextImageEmbedder:
    def __init__(self, text_model_name='all-MiniLM-L6-v2'):
        self.text_model = SentenceTransformer(text_model_name)

    def embed_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.text_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def embed_text_sync(self, text):
        return self.embed_text([text])[0]
