import json
import time
import ssl
import urllib.request
from typing import List, Dict, Any, Optional

try:
    import certifi
    _ssl_context = ssl.create_default_context(cafile=certifi.where())
except Exception:
    _ssl_context = None

from src import config

class EmbeddingEngine:
    """
    Voyage AI Embedding Engine:
    Performs remote inference via Voyage AI API (voyage-3-lite / 512 dimensions).
    Fast, lightweight, and requires ZERO local PyTorch/CUDA dependencies.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or config.VOYAGE_API_KEY
        self.model = model or config.VOYAGE_MODEL
        self.api_url = "https://api.voyageai.com/v1/embeddings"
        self.dimensions = config.EMBEDDING_DIM

    @staticmethod
    def build_search_text(doc: Dict[str, Any]) -> str:
        """Combines catalog text fields into a unified search_text string."""
        tokens = [str(doc[f]).strip() for f in config.SEARCH_TEXT_FIELDS if doc.get(f)]
        return " ".join(tokens)

    def generate_embedding(self, text: str, input_type: str = "query") -> List[float]:
        """Generates a 512-dim normalized embedding vector via Voyage AI API (with auto-retry)."""
        if not text or not text.strip():
            return [0.0] * self.dimensions

        cleaned = text.strip()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": [cleaned],
            "model": self.model,
            "input_type": input_type
        }

        for attempt in range(2):
            try:
                req = urllib.request.Request(
                    self.api_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers
                )
                kwargs = {"timeout": 6}
                if _ssl_context:
                    kwargs["context"] = _ssl_context
                with urllib.request.urlopen(req, **kwargs) as res:
                    if res.status == 200:
                        data = json.loads(res.read().decode("utf-8"))
                        if "data" in data and len(data["data"]) > 0:
                            return data["data"][0]["embedding"]
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt == 0:
                    time.sleep(1.0)
                    continue
                break
            except Exception as e:
                break

        return [0.0] * self.dimensions

    def generate_embeddings_batch(
        self,
        texts: List[str],
        input_type: str = "document",
        batch_size: int = 64
    ) -> List[List[float]]:
        """Generates batch embeddings via Voyage AI API (with auto-retry)."""
        cleaned = [t.strip() if (t and t.strip()) else " " for t in texts]
        if not cleaned:
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        all_embeddings: List[List[float]] = []
        for i in range(0, len(cleaned), batch_size):
            chunk = cleaned[i:i + batch_size]
            payload = {
                "input": chunk,
                "model": self.model,
                "input_type": input_type
            }
            success = False
            for attempt in range(4):
                try:
                    req = urllib.request.Request(
                        self.api_url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers=headers
                    )
                    kwargs = {"timeout": 30}
                    if _ssl_context:
                        kwargs["context"] = _ssl_context
                    with urllib.request.urlopen(req, **kwargs) as res:
                        if res.status == 200:
                            data = json.loads(res.read().decode("utf-8"))
                            if "data" in data:
                                all_embeddings.extend([item["embedding"] for item in data["data"]])
                                success = True
                                break
                except urllib.error.HTTPError as e:
                    if e.code == 429:
                        wait_sec = 2.0 * (attempt + 1)
                        print(f" [Rate-limit 429, backing off {wait_sec}s...] ", end="", flush=True)
                        time.sleep(wait_sec)
                        continue
                    print(f" [HTTP error: {e.code}] ", end="", flush=True)
                    break
                except Exception as e:
                    print(f" [API error: {e}] ", end="", flush=True)
                    time.sleep(1.0)

            if not success:
                all_embeddings.extend([[0.0] * self.dimensions] * len(chunk))

        return all_embeddings

# Singleton instance
_engine: Optional[EmbeddingEngine] = None

def get_embedding_engine() -> EmbeddingEngine:
    """Returns or initializes the global Voyage AI EmbeddingEngine singleton."""
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine()
    return _engine
