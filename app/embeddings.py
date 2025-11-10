from sentence_transformers import SentenceTransformer
import os
import numpy as np


class TextImageEmbedder:
    def __init__(self, text_model_name=None):
        # Automatically load fine-tuned model if available
        default_model = "all-MiniLM-L6-v2"
        tuned_model = os.path.join(os.path.dirname(__file__), "..", "models", "vdoc_feedback_tuned", "latest")

        if text_model_name:
            model_to_use = text_model_name
        elif os.path.exists(os.path.abspath(tuned_model)):
            tuned_path = os.path.abspath(tuned_model)
            print(f"🧠 Using fine-tuned embedding model: {tuned_path}")
            model_to_use = tuned_path
        else:
            print(f"📦 Using base embedding model: {default_model}")
            model_to_use = default_model

        self.text_model = SentenceTransformer(model_to_use)

    def embed_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.text_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def embed_text_sync(self, text):
        return self.embed_text([text])[0]
