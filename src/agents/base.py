import os
import json
from typing import Optional, Dict, Tuple, Any, List, Union
from src import config
from src.search.engine import ProductHybridSearchEngine

# Agent-to-Key Mapping for Rate Limit Distribution
AGENT_KEY_MAP = {
    "query": config.GROQ_API_KEY_QUERY,
    "context": config.GROQ_API_KEY_CONTEXT,
    "validation": config.GROQ_API_KEY_VALIDATION,
    "upsell": config.GROQ_API_KEY_UPSELL,
    "campaign": config.GROQ_API_KEY_CAMPAIGN,
    "default": config.GROQ_API_KEY_QUERY
}

# LLM Cache keyed by (temperature, api_key)
_llm_cache: Dict[Tuple[float, str], Any] = {}

# Real token tracking for monitoring callbacks
_last_token_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

def get_last_token_usage() -> Tuple[int, int]:
    """Returns (prompt_tokens, completion_tokens) from the most recent LLM invocation."""
    return (_last_token_usage.get("prompt_tokens", 0), _last_token_usage.get("completion_tokens", 0))

def set_last_token_usage(prompt_tokens: int, completion_tokens: int):
    """Sets the most recent token count from real API usage metadata."""
    _last_token_usage["prompt_tokens"] = int(prompt_tokens or 0)
    _last_token_usage["completion_tokens"] = int(completion_tokens or 0)


class ChatResponse:
    """Drop-in response wrapper exposing .content attribute matching LangChain's AIMessage."""
    def __init__(self, content: str, usage_metadata: Optional[Dict[str, int]] = None):
        self.content = content
        self.usage_metadata = usage_metadata or {}

    def __str__(self):
        return self.content


class StructuredGroqWrapper:
    """Wrapper implementing with_structured_output for Groq when langchain-groq is unavailable."""
    def __init__(self, parent: "SimpleGroqWrapper", schema: Any):
        self.parent = parent
        self.schema = schema

    def invoke(self, prompt_or_messages: Any) -> Any:
        schema_json = self.schema.schema() if hasattr(self.schema, "schema") else self.schema.model_json_schema()
        system_instruction = (
            "You are an expert AI assistant that strictly returns a JSON object adhering to this schema:\n"
            + json.dumps(schema_json)
        )
        messages = [{"role": "system", "content": system_instruction}]
        if isinstance(prompt_or_messages, str):
            messages.append({"role": "user", "content": prompt_or_messages})
        elif isinstance(prompt_or_messages, list):
            for m in prompt_or_messages:
                if isinstance(m, dict):
                    messages.append(m)
                elif hasattr(m, "content"):
                    role = getattr(m, "type", "user")
                    if role == "human":
                        role = "user"
                    messages.append({"role": role, "content": m.content})

        resp = self.parent.client.chat.completions.create(
            model=self.parent.model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=self.parent.temperature
        )
        if hasattr(resp, "usage") and resp.usage:
            set_last_token_usage(
                getattr(resp.usage, "prompt_tokens", 0),
                getattr(resp.usage, "completion_tokens", 0)
            )
        content = resp.choices[0].message.content
        data = json.loads(content)
        if hasattr(self.schema, "parse_obj"):
            return self.schema.parse_obj(data)
        return self.schema.model_validate(data)


class SimpleGroqWrapper:
    """High-performance direct Groq client wrapper with structured output & chat completion support."""
    def __init__(self, model: str, temperature: float, api_key: str):
        from groq import Groq
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.client = Groq(api_key=api_key)

    def with_structured_output(self, schema: Any) -> StructuredGroqWrapper:
        return StructuredGroqWrapper(self, schema)

    def invoke(self, messages: Any) -> ChatResponse:
        formatted = []
        if isinstance(messages, str):
            formatted.append({"role": "user", "content": messages})
        elif isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict):
                    formatted.append(m)
                elif hasattr(m, "content"):
                    role = getattr(m, "type", "user")
                    if role == "human":
                        role = "user"
                    messages.append({"role": role, "content": m.content})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=formatted,
            temperature=self.temperature
        )
        p_tokens = getattr(resp.usage, "prompt_tokens", 0) if hasattr(resp, "usage") and resp.usage else 0
        c_tokens = getattr(resp.usage, "completion_tokens", 0) if hasattr(resp, "usage") and resp.usage else 0
        set_last_token_usage(p_tokens, c_tokens)
        return ChatResponse(
            resp.choices[0].message.content,
            usage_metadata={"input_tokens": p_tokens, "output_tokens": c_tokens, "total_tokens": p_tokens + c_tokens}
        )


def get_llm(temperature: float = 0.1, agent_name: str = "default") -> Any:
    """
    Returns a ChatGroq LLM instance or SimpleGroqWrapper configured with dedicated API key per agent.
    - Query Agent -> Key 1
    - Context & Validation Agents -> Key 2
    - Upsell Stylist Agent -> Key 3
    """
    global _llm_cache
    api_key = AGENT_KEY_MAP.get(agent_name, config.GROQ_API_KEY_QUERY)
    if not api_key:
        api_key = config.GROQ_API_KEY_QUERY or config.GROQ_API_KEY

    cache_key = (round(temperature, 2), api_key)
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    try:
        from langchain_groq import ChatGroq
        instance = ChatGroq(
            model=config.LLM_MODEL,
            temperature=temperature,
            api_key=api_key
        )
    except Exception:
        instance = SimpleGroqWrapper(
            model=config.LLM_MODEL,
            temperature=temperature,
            api_key=api_key
        )

    _llm_cache[cache_key] = instance
    return instance


# Lazy initialization of Search Engine
_search_engine: Optional[ProductHybridSearchEngine] = None

def get_search_engine() -> ProductHybridSearchEngine:
    """Returns the shared ProductHybridSearchEngine instance (lazy initialized)."""
    global _search_engine
    if _search_engine is None:
        _search_engine = ProductHybridSearchEngine()
    return _search_engine
