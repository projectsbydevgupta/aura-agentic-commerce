from typing import Dict, Any, List, Optional
import re

INR_TO_USD = 50.0  # 1 USD = ₹50 INR

# Threshold: any price >300 USD is almost certainly INR-scale in our fashion catalog context
# (catalog items range from ~$15 to ~$400 USD; INR values are 750–20000)
USD_CATALOG_MAX_REALISTIC = 300.0

def _detect_inr_in_query(query: str) -> bool:
    """Returns True if query text contains INR indicators."""
    if not query:
        return False
    inr_patterns = [
        r'₹', r'\brs\.?\b', r'\binr\b', r'\brupee', r'\blakh\b', r'\bk\s*rs\b'
    ]
    q_lower = query.lower()
    return any(re.search(p, q_lower) for p in inr_patterns)

def normalize_price_to_usd(price: Optional[float], user_query: str = "") -> Optional[float]:
    """
    Deterministic post-LLM price sanity check.
    If the LLM output a price that looks like INR (>300 and query contains INR signals,
    OR >600 regardless — no fashion item in catalog costs >$400 USD),
    divide by INR_TO_USD to convert to catalog USD units.
    """
    if price is None:
        return None
    # If price is obviously INR-scale: either user mentioned ₹/INR, or the number itself is too large
    if price > USD_CATALOG_MAX_REALISTIC:
        return round(price / INR_TO_USD, 2)
    # If query text clearly has INR markers and price looks like a round INR amount
    if _detect_inr_in_query(user_query) and price > 100:
        return round(price / INR_TO_USD, 2)
    return price

def normalize_filter_prices(filter_dict: Dict[str, Any], user_query: str = "") -> Dict[str, Any]:
    """
    Walks a filter dict and normalizes any price values from INR to USD if needed.
    Handles nested price dicts like {'price': {'$lte': 3000}}.
    """
    if not filter_dict:
        return filter_dict
    result = dict(filter_dict)
    if 'price' in result and isinstance(result['price'], dict):
        p = dict(result['price'])
        if '$lte' in p:
            p['$lte'] = normalize_price_to_usd(p['$lte'], user_query)
        if '$gte' in p:
            p['$gte'] = normalize_price_to_usd(p['$gte'], user_query)
        result['price'] = p
    # Also handle top-level min_price / max_price keys (sometimes LLM puts them here)
    if 'max_price' in result:
        result['max_price'] = normalize_price_to_usd(result['max_price'], user_query)
    if 'min_price' in result:
        result['min_price'] = normalize_price_to_usd(result['min_price'], user_query)
    return result

