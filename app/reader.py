import os
from typing import List, Dict
from dotenv import dotenv_values
from transformers import pipeline

# --- Load ONLY from .env (ignore system environment) ---
env_vars = dotenv_values(".env")  # returns a dict from the .env file

# --- FIX: Use the correct modern SDK import (google-genai) and initialize client ---
genai = None
_gemini_client = None
try:
    from google import genai
    from google.genai import types
    
    # Initialize client (the new SDK automatically reads GEMINI_API_KEY if set)
    # We'll inject it manually from env_vars to avoid system env dependency
    _gemini_client = genai.Client(api_key=env_vars.get("GEMINI_API_KEY"))
except ImportError:
    pass
except Exception as e:
    print(f"Warning: Failed to initialize Gemini client. Check API key/configuration. Error: {e}")

class LLMReader:
    """
    LLM Reader using Google Gemini (via GEMINI_API_KEY from .env only)
    Falls back to a local small model if unavailable.
    """

    def __init__(self, provider: str = "gemini"):
        self.provider = provider.lower()
        
        # Load only from .env
        self.model = env_vars.get("VDOCRAG_LLM_MODEL", "gemini-2.5-flash")
        self.api_key = env_vars.get("GEMINI_API_KEY")
        self.client = _gemini_client
        self.local_pipeline = None

        print("=" * 50)
        print(f"LLMReader Init: Loading GEMINI_API_KEY from .env only...")
        if self.api_key:
            print(f"LLMReader Init: SUCCESS. Key prefix: {self.api_key[:4]}...{self.api_key[-4:]}")
        else:
            print(f"LLMReader Init: FAILED. GEMINI_API_KEY not found in .env.")
        print("=" * 50)

        if self.provider == "gemini":
            if self.client is None:
                raise ImportError("Please install the modern Google GenAI SDK: `pip install google-genai`.")
            if not self.api_key:
                print("⚠️ No GEMINI_API_KEY found in .env, switching to local model.")
                self.provider = "local"

        if self.provider == "local":
            print(f"Loading local model: distilgpt2...")
            self.local_pipeline = pipeline("text-generation", model="distilgpt2")

        if self.provider not in ("gemini", "local"):
            print(f"⚠️ Unknown provider '{self.provider}', defaulting to local.")
            self.provider = "local"
            if self.local_pipeline is None:
                print(f"Loading local model: distilgpt2...")
                self.local_pipeline = pipeline("text-generation", model="distilgpt2")

    # --------------------------
    # Gemini call (modern SDK)
    # --------------------------
    def _call_gemini(self, query: str, context: str) -> str:
        system_prompt = (
            "You are a precise data analysis assistant. "
            "Given the provided CONTEXT, answer the user's QUESTION accurately. "
            "If calculations are needed, perform them. "
            "Only respond with the final answer and no additional commentary or explanation."
        )

        user_content = f"CONTEXT:\n---\n{context}\n---\nQUESTION: {query}"

        try:
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_content,
                config=config
            )
            return response.text.strip()
        except Exception as e:
            return f"[Gemini API Error] {type(e).__name__}: {e}"

    # --------------------------
    # Local fallback
    # --------------------------
    def _call_local(self, query: str, context: str) -> str:
        prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"Based on the context, answer the following question:\n"
            f"QUESTION: {query}\n"
            f"ANSWER:"
        )

        result = self.local_pipeline(
            prompt,
            max_new_tokens=100,
            do_sample=True,
            truncation=True
        )
        generated_text = result[0]["generated_text"]
        answer = generated_text[len(prompt):].strip()

        if not answer or context in answer:
            return "[Local model failed to generate a new answer and may have repeated the context]"
        return answer

    # --------------------------
    # Main answer method
    # --------------------------
    def answer_question(self, query: str, context: str, sources: List[Dict]) -> Dict:
        if self.provider == "gemini":
            answer_text = self._call_gemini(query, context)
        elif self.provider == "local":
            answer_text = self._call_local(query, context)
        else:
            answer_text = f"[Error: Unknown provider '{self.provider}']"

        provenance = [
            {
                "page": s["metadata"].get("page"),
                "text": s["text"][:200],
                "score": s.get("score", 0),
            }
            for s in sources
        ]

        return {"text": answer_text, "sources": provenance}
