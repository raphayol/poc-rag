import requests
import os
from dotenv import load_dotenv

load_dotenv()


class OllamaClient:
    """Handles communication with the Ollama API."""

    def __init__(self):
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.generation_model = os.getenv("OLLAMA_GENERATION_MODEL", "deepseek-r1:1.5b")

    def _call_api(self, endpoint: str, payload: dict) -> dict:
        r = requests.post(
            f"{self.url}/api/{endpoint}",
            json=payload,
            timeout=120
        )
        r.raise_for_status()

        return r.json()

    def embed(self, text: str) -> list[float]:
        """Generates a single embedding vector."""
        payload = {"model": self.embedding_model, "prompt": text}

        return self._call_api("embeddings", payload)["embedding"]

    def generate(self, prompt: str) -> str:
        """Generates a response from the LLM."""
        payload = {"model": self.generation_model, "prompt": prompt, "stream": False}

        return self._call_api("generate", payload)["response"]
