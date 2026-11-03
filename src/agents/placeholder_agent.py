import json
import random
from typing import List, Dict, Any, Optional
from src.agents.base import get_llm

# Curated High-Fashion & Luxury Catalog Prompts
DEFAULT_CATALOG_PROMPTS = [
    "Minimal black running sneakers under $80...",
    "Blue casual cotton shirt for dinner...",
    "White summer linen dress under $60...",
    "Men slim fit black formal trousers...",
    "Classic silver chronograph watch under $100...",
    "Navy blue hooded sweatshirt for winter...",
    "Women casual floral print top under $45...",
    "Puma breathable sports running shoes...",
    "Beige tailored semi-formal blazer...",
    "Black leather crossbody handbag...",
    "Olive green casual cargo pants...",
    "Red round neck cotton t-shirt under $30...",
    "Brown formal leather shoes for office...",
    "Unisex vintage black sunglasses...",
    "Monochrome street luxury hoodie under $90...",
    "Silk slip evening midi dress in champagne...",
    "Minimalist gold accent wristwatch for women...",
    "Italian calfskin oxford dress shoes...",
    "Structured double-breasted charcoal overcoat...",
    "High-waisted tailored wide leg trousers...",
    "Textured knit cashmere sweater in oatmeal...",
    "Leather Chelsea boots in espresso brown..."
]

# Dynamic, Engaging Pipeline Thoughts (Replaces Dry Static Text)
DYNAMIC_PIPELINE_SETS = [
    [
        {"step": 1, "agent": "Query Agent", "thought": "Parsing modern athleisure & tailoring specifications..."},
        {"step": 2, "agent": "Context Agent", "thought": "Filtering occasion boundaries & climate appropriateness..."},
        {"step": 3, "agent": "Search Node", "thought": "Querying high-dimensional embeddings across MongoDB Atlas..."},
        {"step": 4, "agent": "Validation Agent", "thought": "Screening top recommendations against patron fit score..."},
        {"step": 5, "agent": "Stylist Agent", "thought": "Polishing the curated ensemble with bespoke designer tips..."}
    ],
    [
        {"step": 1, "agent": "Query Agent", "thought": "Decoding silhouette proportions & aesthetic vibe..."},
        {"step": 2, "agent": "Context Agent", "thought": "Cross-referencing runway trends & seasonal occasion..."},
        {"step": 3, "agent": "Search Node", "thought": "Deep-mining 44,000+ luxury pieces with Voyage AI 512-dim vectors..."},
        {"step": 4, "agent": "Validation Agent", "thought": "Quality-assuring fabric drape, color harmony & rating score..."},
        {"step": 5, "agent": "Stylist Agent", "thought": "Curating bespoke capsule pairings & editorial styling advice..."}
    ],
    [
        {"step": 1, "agent": "Query Agent", "thought": "Dissecting streetwear drip, fit profile & color harmony..."},
        {"step": 2, "agent": "Context Agent", "thought": "Analyzing subtle style constraints & price-to-luxury ratio..."},
        {"step": 3, "agent": "Search Node", "thought": "Fusing Atlas Vector Search + Keyword Rank Fusion (RRF)..."},
        {"step": 4, "agent": "Validation Agent", "thought": "Inspecting piece compatibility & boutique in-stock status..."},
        {"step": 5, "agent": "Stylist Agent", "thought": "Synthesizing Haute Couture styling notes for complete look..."}
    ],
    [
        {"step": 1, "agent": "Query Agent", "thought": "Calibrating clean-girl & quiet luxury aesthetic parameters..."},
        {"step": 2, "agent": "Context Agent", "thought": "Evaluating versatility for effortless day-to-night transitions..."},
        {"step": 3, "agent": "Search Node", "thought": "Scanning catalog archive for high-affinity editorial pieces..."},
        {"step": 4, "agent": "Validation Agent", "thought": "Verifying brand craftsmanship & silhouette integrity..."},
        {"step": 5, "agent": "Stylist Agent", "thought": "Composing matching footwear & accessory tonal accents..."}
    ]
]


class PlaceholderAgent:
    """
    Dynamic Search Placeholder & Progress Guidance Agent.
    Provides curated luxury fashion search prompts, dynamic pipeline thoughts,
    and editorial combo styling suggestions.
    """

    def __init__(self, model: str = "openai/gpt-oss-20b"):
        self.model = model
        self._prompt_pool: List[str] = list(DEFAULT_CATALOG_PROMPTS)
        random.shuffle(self._prompt_pool)
        self._pool_index: int = 0
        self._pipeline_set_index: int = 0

    def get_next_prompt(self) -> str:
        """Returns the next prompt from the active pool, rotating smoothly."""
        if not self._prompt_pool:
            self._prompt_pool = list(DEFAULT_CATALOG_PROMPTS)
            random.shuffle(self._prompt_pool)

        prompt = self._prompt_pool[self._pool_index % len(self._prompt_pool)]
        self._pool_index += 1
        return prompt

    def get_dynamic_pipeline_steps(self) -> List[Dict[str, Any]]:
        """Returns a dynamic, engaging set of agent thoughts for the progressive revelation banner."""
        steps_set = DYNAMIC_PIPELINE_SETS[self._pipeline_set_index % len(DYNAMIC_PIPELINE_SETS)]
        self._pipeline_set_index += 1
        return steps_set

    def generate_fresh_batch(self, count: int = 6) -> List[str]:
        """Returns a fresh randomized batch of curated luxury fashion prompts."""
        selected = random.sample(DEFAULT_CATALOG_PROMPTS, min(count, len(DEFAULT_CATALOG_PROMPTS)))
        return selected

    def generate_combo_suggestion(self, anchor: Dict[str, Any], pairings: List[Dict[str, Any]]) -> str:
        """
        Synthesizes a chic, tailored styling combo tip incorporating the anchor piece
        and recommended complementary pieces.
        """
        anchor_name = anchor.get("name", "Anchor Piece")
        anchor_brand = anchor.get("brand", "")
        anchor_type = anchor.get("article_type", "apparel")
        anchor_color = anchor.get("base_color", "")

        pair_names = [p.get("name", "") for p in pairings if p.get("name")]

        # Tonal context-aware fallback
        if len(pair_names) >= 3:
            fallback_tip = f"Pair the {anchor_name} with {pair_names[0]} and {pair_names[1]}, grounded by {pair_names[2]} for an impeccable tonal aesthetic."
        elif len(pair_names) >= 2:
            fallback_tip = f"Style the {anchor_name} seamlessly alongside {pair_names[0]} and {pair_names[1]} for an elevated, harmonious look."
        elif len(pair_names) == 1:
            fallback_tip = f"Match the {anchor_name} directly with {pair_names[0]} for a balanced, effortlessly curated silhouette."
        else:
            fallback_tip = f"Ground this {anchor_name} with tailored minimalist tones; keep the lines effortless and refined."

        try:
            llm = get_llm(temperature=0.7, agent_name="stylist")
            pair_summary = ", ".join(pair_names[:3])
            prompt = (
                f"You are AURA's Elite Runway Stylist. Write 1-2 concise, elegant sentences on how to wear "
                f"this look together: Anchor is {anchor_name} ({anchor_color} {anchor_type} by {anchor_brand}), "
                f"paired with {pair_summary}. Focus on drape, tonal harmony, and occasion elegance. No bullet points."
            )
            response = llm.invoke(prompt)
            content = response.content.strip() if hasattr(response, "content") else str(response).strip()
            if content and len(content) > 15:
                return content.replace('"', '').strip()
        except Exception as e:
            print(f"Notice on combo suggestion generation fallback: {e}")

        return fallback_tip


# Global singleton instance
_placeholder_agent = None

def get_placeholder_agent() -> PlaceholderAgent:
    global _placeholder_agent
    if _placeholder_agent is None:
        _placeholder_agent = PlaceholderAgent()
    return _placeholder_agent
