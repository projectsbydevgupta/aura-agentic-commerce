import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
"""
Populates data/monitoring_logs with real agent telemetry traffic:
1. 1 real Campaign Agent copy generation (logged to monitoring JSONL)
2. 20 real /api/chat requests executed through the LangGraph agent pipeline
"""
import asyncio
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server import handle_chat, ChatRequest
from src.agents.campaign_agent import get_campaign_agent

CHAT_PROMPTS = [
    "men black formal shoes under 4000",
    "women floral summer dress under 3000",
    "help me find an outfit for a summer wedding",
    "white linen shirt for men",
    "luxury brown leather handbag",
    "cushioned running sneakers for marathon",
    "minimalist gold link bracelet accessories",
    "oversized aesthetic hoodie in beige",
    "classic blue denim jacket for women",
    "what should I wear to a cocktail evening party",
    "casual polo t-shirt in forest green",
    "dark polarized aviator sunglasses",
    "tailored black formal trousers for men",
    "comfortable walking leather flats",
    "breathable sports gym track pants",
    "silk evening scarf with handloom print",
    "stainless steel chronograph wristwatch",
    "leather ankle boots for autumn",
    "bespoke formal dinner blazer",
    "metallic silver stiletto evening heels"
]

async def main():
    print("=" * 60)
    print("  STEP 1: Generating 1 Real Campaign Agent Message")
    print("=" * 60)
    camp = get_campaign_agent()
    camp_res = camp.generate_message(
        customer_name="Aarav Sharma",
        bag_items=[
            {"name": "Midnight Velvet Blazer", "price": 120.0, "article_type": "Blazers"},
            {"name": "Silk Pocket Square", "price": 25.0, "article_type": "Accessories"}
        ],
        channel="whatsapp",
        discount_code="AURA20",
        tone="witty_hinglish"
    )
    print(f"✓ Campaign Agent generated headline: {camp_res.get('headline')}")

    print("\n" + "=" * 60)
    print(f"  STEP 2: Executing 20 Real /api/chat Requests")
    print("=" * 60)
    for i, prompt in enumerate(CHAT_PROMPTS, 1):
        print(f"[{i:02d}/20] Processing query: '{prompt}'...")
        req = ChatRequest(message=prompt, thread_id=f"traffic_thread_{i:02d}")
        try:
            res = await handle_chat(req)
            intent = res.get("intent", "unknown")
            results_count = len(res.get("search_results", []))
            upsell_count = len(res.get("upsell_results", []))
            print(f"       -> Intent: {intent} | Items: {results_count} | Upsells: {upsell_count}")
        except Exception as e:
            print(f"       -> Error: {e}")

    print("\n✓ Finished 20 real chat requests and 1 campaign generation.")

if __name__ == "__main__":
    asyncio.run(main())
