import os
from typing import List, Dict
from dotenv import load_dotenv
from transformers import pipeline

# Load environment variables from .env file
load_dotenv()

# --- FIX: Use the correct modern SDK import (google-genai) and initialize client ---
genai = None
_gemini_client = None
try:
    # 1. Correct Import: The new library uses 'from google import genai'
    from google import genai
    from google.genai import types
    
    # 2. Correct Initialization: The client object is required for all calls
    # It automatically reads GEMINI_API_KEY from the environment
    _gemini_client = genai.Client()
except ImportError:
    # This block handles if 'google-genai' is not installed
    pass
except Exception as e:
    print(f"Warning: Failed to initialize Gemini client. Check API key/configuration. Error: {e}")

class LLMReader:
    """
    LLM Reader using Google Gemini (via GEMINI_API_KEY)
    Falls back to local small LLM if the SDK is unavailable or the key is missing.
    """

    def __init__(self, provider: str = "gemini"):
        self.provider = provider.lower()
        # Use a model that supports system instructions well
        self.model = os.environ.get("VDOCRAG_LLM_MODEL", "gemini-2.5-flash")
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.client = _gemini_client
        self.local_pipeline = None  # <-- FIX: Initialize to None

        # --- DIAGNOSTIC PRINT ---
        # Add this print statement to check what's being loaded.
        print("="*50)
        print(f"LLMReader Init: Attempting to load GEMINI_API_KEY...")
        if self.api_key:
            # Show only the first 4 and last 4 chars so the key isn't exposed
            print(f"LLMReader Init: SUCCESS. Found key: {self.api_key[:4]}...{self.api_key[-4:]}")
        else:
            print(f"LLMReader Init: FAILED. GEMINI_API_KEY not found in environment.")
        print("="*50)

        if self.provider == "gemini":
            if self.client is None:
                # FIX: Updated package name in error message
                raise ImportError("Please install the modern Google GenAI SDK: `pip install google-genai`.")

            if not self.api_key:
                # The client might be initialized, but we check the key existence for the user notification
                print("⚠️ GEMINI_API_KEY missing in .env, switching to local model.")
                self.provider = "local"
                # Fall-through to the 'local' block
            
            # REMOVED: genai.configure() is deprecated in the modern SDK.

        # Local fallback
        if self.provider == "local":
            print(f"Loading local model: distilgpt2...")
            # distilgpt2 is small and fast for a local fallback
            self.local_pipeline = pipeline("text-generation", model="distilgpt2")
        
        # <-- FIX: Add a final check to catch invalid providers and default to local -->
        if self.provider not in ("gemini", "local"):
            print(f"⚠️ Unknown provider '{self.provider}', defaulting to local model.")
            self.provider = "local"
            if self.local_pipeline is None: # Only load if not already loaded
                print(f"Loading local model: distilgpt2...")
                self.local_pipeline = pipeline("text-generation", model="distilgpt2")

    # --------------------------
    # Gemini call (using modern SDK and system instructions)
    # --------------------------
    def _call_gemini(self, query: str, context: str) -> str:
        """
        Use Gemini API to generate an answer with reasoning, leveraging system instructions.
        """
        
        # System instructions define the model's behavior for the RAG task
        system_prompt = (
            "You are a precise data analysis assistant. "
            "Given the provided CONTEXT, answer the user's QUESTION accurately. "
            "If calculations are needed, perform them. "
            "Only respond with the final answer and no additional commentary or explanation."
        )

        # The user content combines the context and the final question
        user_content = (
            f"CONTEXT:\n---\n{context}\n---\n"
            f"QUESTION: {query}"
        )

        try:
            # Use the dedicated system_instruction parameter and low temperature for accuracy
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1 # Low temperature prioritizes factual consistency for RAG
            )
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_content,
                config=config
            )

            return response.text.strip()

        except Exception as e:
            # Better error reporting
            return f"[Gemini API Error] {type(e).__name__}: {e}"

    # --------------------------
    # Local fallback
    # --------------------------
    def _call_local(self, query: str, context: str) -> str:
        # Concatenate context and query for the local model
        # --- FIX: Re-ordered prompt for better instruction-following ---
        # Put the context first, then the explicit instruction and question.
        # This is more likely to work with a base text-generation model.
        prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"Based on the context, answer the following question:\n"
            f"QUESTION: {query}\n"
            f"ANSWER:"
        )
        
        # Ensure the prompt fits the small model's context window (max_length)
        # We use truncation=True to prevent errors when inputs are too long
        # --- FIX: Use max_new_tokens to give space for the answer ---
        # max_length=200 was cutting off the answer, as it included the prompt length.
        # max_new_tokens generates 100 tokens *after* the prompt.
        result = self.local_pipeline(
            prompt, 
            max_new_tokens=100, # Generate 100 new tokens for the answer
            do_sample=True, 
            truncation=True    # Truncate the prompt if it's > 1024 tokens
        )
        
        # Clean up the output by removing the input prompt
        generated_text = result[0]["generated_text"]
        
        # --- FIX: Use a more robust way to get only the new text ---
        # .replace() can fail if the model adds a space or newline
        answer = generated_text[len(prompt):].strip()
        
        # The model might just generate an empty string.
        # If the answer is empty or just repeats the context, return a clearer error.
        if not answer or context in answer:
             return "[Local model failed to generate a new answer and may have repeated the context]"
        
        return answer


    # --------------------------
    # Main
    # --------------------------
    def answer_question(self, query: str, context: str, sources: List[Dict]) -> Dict:
        if self.provider == "gemini":
            answer_text = self._call_gemini(query, context)
        elif self.provider == "local":  # <-- FIX: Make this an explicit 'elif'
            answer_text = self._call_local(query, context)
        else:
            # This should no longer be reachable due to the __init__ fix,
            # but it's good defensive programming.
            answer_text = f"[Error: Unknown provider '{self.provider}'. No model was called.]"

        # Process source metadata (provenance)
        provenance = [
            {
                "page": s["metadata"].get("page"),
                "text": s["text"][:200],
                "score": s.get("score", 0),
            }
            for s in sources
        ]
        
        return {"text": answer_text, "sources": provenance}