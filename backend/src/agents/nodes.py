"""
AURA - Instrumented Agent Nodes for LangGraph
Implements telemetry instrumentation wrapping Query, Context, Search, Validation, and Upsell nodes
with on_agent_start, on_agent_end, on_agent_error from src.monitoring.callbacks, capturing real latency
and real token usage from LLM API responses.
"""
import time
import uuid
from typing import Dict, Any

from src.agents.base import get_llm, get_search_engine, get_last_token_usage, set_last_token_usage
from src.monitoring.callbacks import get_monitor_callback
from src.agents.query_agent import query_agent_node as _raw_query_agent_node
from src.agents.context_agent import context_agent_node as _raw_context_agent_node
from src.agents.search_agent import search_node as _raw_search_node
from src.agents.validation_agent import validation_agent_node as _raw_validation_agent_node
from src.agents.upsell_agent import upsell_agent_node as _raw_upsell_agent_node, COMPLEMENTARY_MAP


def instrumented_query_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    monitor = get_monitor_callback()
    thread_id = state.get("thread_id", "default_thread")
    step_id = f"query_{uuid.uuid4().hex[:8]}"
    input_text = state.get("current_query") or state.get("original_query", "")
    monitor.on_agent_start(step_id, thread_id, "Query Agent", input_text)
    set_last_token_usage(0, 0)
    try:
        res = _raw_query_agent_node(state)
        p_tokens, c_tokens = get_last_token_usage()
        output_summary = f"Intent: {res.get('intent')}, Filters: {res.get('filters', {})}"
        monitor.on_agent_end(step_id, output_summary, prompt_tokens=p_tokens, completion_tokens=c_tokens)
        return res
    except Exception as e:
        monitor.on_agent_error(step_id, e)
        raise


def instrumented_context_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    monitor = get_monitor_callback()
    thread_id = state.get("thread_id", "default_thread")
    step_id = f"context_{uuid.uuid4().hex[:8]}"
    input_text = state.get("current_query", "")
    monitor.on_agent_start(step_id, thread_id, "Context Agent", input_text)
    set_last_token_usage(0, 0)
    try:
        res = _raw_context_agent_node(state)
        p_tokens, c_tokens = get_last_token_usage()
        output_summary = f"Clarify: {res.get('needs_clarification')}, Query: {res.get('current_query')}"
        monitor.on_agent_end(step_id, output_summary, prompt_tokens=p_tokens, completion_tokens=c_tokens)
        return res
    except Exception as e:
        monitor.on_agent_error(step_id, e)
        raise


def instrumented_search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    monitor = get_monitor_callback()
    thread_id = state.get("thread_id", "default_thread")
    step_id = f"search_{uuid.uuid4().hex[:8]}"
    input_text = f"Query: {state.get('current_query')}, Filters: {state.get('filters')}"
    monitor.on_agent_start(step_id, thread_id, "Search Node", input_text)
    t0 = time.perf_counter()
    try:
        res = _raw_search_node(state)
        dur_ms = round((time.perf_counter() - t0) * 1000, 2)
        results = res.get("search_results", [])
        monitor.on_tool_call(step_id, "catalog_hybrid_search", state.get("current_query"), f"Found {len(results)} items", dur_ms)
        # Search is vector/keyword search (0 LLM tokens)
        monitor.on_agent_end(step_id, f"Retrieved {len(results)} products", prompt_tokens=0, completion_tokens=0)
        return res
    except Exception as e:
        monitor.on_agent_error(step_id, e)
        raise


def instrumented_validation_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    monitor = get_monitor_callback()
    thread_id = state.get("thread_id", "default_thread")
    step_id = f"validation_{uuid.uuid4().hex[:8]}"
    input_text = f"Items: {len(state.get('search_results', []))}, Query: {state.get('original_query')}"
    monitor.on_agent_start(step_id, thread_id, "Validation Agent", input_text)
    set_last_token_usage(0, 0)
    try:
        res = _raw_validation_agent_node(state)
        p_tokens, c_tokens = get_last_token_usage()
        v_res = res.get("validation_result", {})
        output_summary = f"Validated: {v_res.get('validated')}, Reason: {v_res.get('explanation', '')[:60]}"
        monitor.on_agent_end(step_id, output_summary, prompt_tokens=p_tokens, completion_tokens=c_tokens)
        return res
    except Exception as e:
        monitor.on_agent_error(step_id, e)
        raise


def instrumented_upsell_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    monitor = get_monitor_callback()
    thread_id = state.get("thread_id", "default_thread")
    step_id = f"upsell_{uuid.uuid4().hex[:8]}"
    anchor = state.get("selected_product") or (state.get("search_results") or [{}])[0]
    input_text = f"Anchor: {anchor.get('name', 'None')}"
    monitor.on_agent_start(step_id, thread_id, "Upsell Agent", input_text)
    set_last_token_usage(0, 0)
    try:
        res = _raw_upsell_agent_node(state)
        p_tokens, c_tokens = get_last_token_usage()
        upsells = res.get("upsell_results", [])
        output_summary = f"Recommended {len(upsells)} outfit pieces"
        monitor.on_agent_end(step_id, output_summary, prompt_tokens=p_tokens, completion_tokens=c_tokens)
        return res
    except Exception as e:
        monitor.on_agent_error(step_id, e)
        raise


# Re-export instrumented nodes as primary exports
query_agent_node = instrumented_query_agent_node
context_agent_node = instrumented_context_agent_node
search_node = instrumented_search_node
validation_agent_node = instrumented_validation_agent_node
upsell_agent_node = instrumented_upsell_agent_node

__all__ = [
    "get_llm",
    "get_search_engine",
    "query_agent_node",
    "context_agent_node",
    "search_node",
    "validation_agent_node",
    "upsell_agent_node",
    "COMPLEMENTARY_MAP"
]
