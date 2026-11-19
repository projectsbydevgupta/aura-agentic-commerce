from typing import Dict, Any
from src.agents.state import AgentState, ContextEvaluation
from src.agents.base import get_llm
from src.agents.price_utils import normalize_filter_prices

def context_agent_node(state: AgentState) -> Dict[str, Any]:
    """
    Context Agent: Evaluator & Clarification Generator
    Evaluates whether enough information exists to execute a targeted search.
    If critically vague / under-specified and clarification_count < 2, asks 1 concise question.
    """
    clarification_count = state.get("clarification_count", 0)
    current_query = state.get("current_query", "")
    original_query = state.get("original_query", current_query)
    filters = state.get("filters", {})

    if state.get("selected_product") and state.get("intent") == "select_product":
        return {
            "needs_clarification": False,
            "clarification_question": None,
            "current_query": current_query,
            "filters": filters
        }

    if clarification_count >= 2:
        return {
            "needs_clarification": False,
            "clarification_question": None,
            "current_query": current_query,
            "filters": filters
        }

    history = state.get("conversation_history", [])
    dialogue_context = ""
    if history:
        formatted = [f"{t.get('role', 'user').capitalize()}: {t.get('content', '')}" for t in history[-4:]]
        dialogue_context = "Recent Conversation History:\n" + "\n".join(formatted) + "\n\n"

    prompt = f"""You are an Expert Conversational E-Commerce Fashion Context Evaluator for AURA Luxury Concierge.
Determine if the shopping request has sufficient specificity to execute a targeted catalog search, or if it is ambiguous/under-specified and requires clarifying details from the user.

{dialogue_context}Original User Message: "{original_query}"
Current Search Term: "{current_query}"
Current Filters: {filters}
Clarification Rounds So Far: {clarification_count}/2

Evaluation Principles:
1. Under-Specified / Ambiguous Requests (needs clarification):
   - If the request is generic, broad, or mentions only a broad product category WITHOUT qualifiers (e.g. "shoes", "i want to buy a shoes", "sneakers", "clothes", "watch", "shirt", "dress", "something to wear", "give me clothes"):
     It lacks critical dimensions: target gender/audience (Men vs. Women vs. Kids), style/occasion (formal, casual, sports), or budget.
     -> Set has_sufficient_context = False.
     -> Generate a concise, polite luxury concierge clarification question to narrow down the missing details (e.g. "Sure! Are you looking for men's or women's shoes? Also, what type of shoes do you need – formal, casual or sports?").

2. Sufficient Context Requests (proceed directly to search):
   - If the user provides specific qualifiers (e.g. gender: "men shoes", "women dresses"; OR style/occasion: "formal shoes", "running shoes", "party wear"; OR brand: "Nike sneakers"; OR budget: "shoes under ₹3000", "blazer under 8000"):
     -> Set has_sufficient_context = True.
     -> Set clarification_question = None.
     -> Set final_search_query to the refined search string.

3. Answering Clarification:
   - If the user's message is an answer to a previous clarification (e.g. "Men's formal shoes under ₹5,000", "for men", "casual"):
     -> Incorporate the new information into final_search_query and inferred_filters.
     -> Set has_sufficient_context = True.
     -> Set clarification_question = None.

4. Dynamic Clarification Options (Quick Suggestions):
   - When has_sufficient_context = False, you MUST generate 3 concise quick suggestion options in clarification_options tailored specifically to the user's inquiry and item category with realistic budgets in ₹ INR (e.g., for shirts: ["Men's formal under ₹2,500", "Women's casual under ₹2,000", "Men's linen under ₹3,000"]; for shoes: ["Men's formal under ₹5,000", "Women's casual under ₹4,000", "Men's running under ₹3,000"]; for dresses: ["Women's party wear under ₹5,000", "Women's casual floral under ₹3,500", "Women's maxi under ₹4,500"]; for shorts: ["Men's denim under ₹2,000", "Women's casual under ₹1,800", "Men's athletic under ₹1,500"]; for watches: ["Men's analog under ₹6,000", "Women's rose gold under ₹5,000", "Unisex chronograph under ₹8,000"]).
"""
    llm = get_llm(temperature=0.1, agent_name="context")
    structured_llm = llm.with_structured_output(ContextEvaluation)
    try:
        eval_result: ContextEvaluation = structured_llm.invoke(prompt)
    except Exception as e:
        print(f"Context evaluation notice: {e}")
        has_ctx = bool(filters or len(current_query.split()) >= 3)
        eval_result = ContextEvaluation(
            has_sufficient_context=has_ctx,
            clarification_question=None if has_ctx else "Could you share your preferred style, occasion, or budget for this piece?",
            clarification_options=[] if has_ctx else [f"Men's casual under ₹3,000", f"Women's casual under ₹2,500"],
            final_search_query=current_query,
            inferred_filters=filters
        )

    updated_history = list(history)

    if not eval_result.has_sufficient_context and clarification_count < 2:
        if eval_result.clarification_question:
            updated_history.append({"role": "assistant", "content": eval_result.clarification_question})

        # Dynamic options tailored specifically to the user's requested item
        dynamic_opts = list(eval_result.clarification_options or [])
        if not dynamic_opts:
            raw_item = (current_query or original_query or "pieces").lower()
            clean_item = raw_item.replace("i want to buy", "").replace("i need", "").replace("find", "").replace("show me", "").replace("a ", "").replace("an ", "").strip()
            dynamic_opts = [
                f"Men's formal {clean_item} under ₹3,500",
                f"Women's casual {clean_item} under ₹3,000",
                f"Men's casual {clean_item} under ₹2,500"
            ]

        return {
            "needs_clarification": True,
            "clarification_question": eval_result.clarification_question,
            "clarification_options": dynamic_opts,
            "clarification_count": clarification_count + 1,
            "conversation_history": updated_history
        }
    else:
        merged = filters.copy()
        if eval_result.inferred_filters:
            # Normalize any prices in inferred_filters from INR to USD catalog units
            safe_inferred = normalize_filter_prices(eval_result.inferred_filters, original_query)
            if safe_inferred != eval_result.inferred_filters:
                print(f"[ContextAgent] inferred_filters price normalized: {eval_result.inferred_filters} -> {safe_inferred}")
            merged.update(safe_inferred)
        # Also normalize any prices already in merged filters (from query agent)
        merged = normalize_filter_prices(merged, original_query)

        return {
            "needs_clarification": False,
            "clarification_question": None,
            "current_query": eval_result.final_search_query or current_query,
            "filters": merged,
            "conversation_history": updated_history
        }
