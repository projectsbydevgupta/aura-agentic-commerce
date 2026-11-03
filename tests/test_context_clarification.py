"""
AURA Context Agent & Clarification Flow Unit Tests
Tests clarification behavior for vague vs. specific user queries,
as well as dynamic suggestion options and context progression.
"""

import pytest
from src.agents.workflow import agent_app


def test_vague_query_clarification():
    """Vague queries (e.g. 'I need shoes') must request clarification with dynamic options."""
    thread_id = "test_vague_shoes_clarification"
    config = {"configurable": {"thread_id": thread_id}}
    state = {
        "current_query": "I need shoes",
        "original_query": "I need shoes",
        "search_results": [],
        "selected_product": None,
        "upsell_results": [],
        "clarification_count": 0,
    }
    result = agent_app.invoke(state, config=config)
    assert result.get("needs_clarification") is True
    assert result.get("clarification_question") is not None
    assert len(result.get("search_results", [])) == 0
    # Must provide dynamic clarification options
    options = result.get("clarification_options", [])
    assert isinstance(options, list)
    assert len(options) > 0


def test_specific_query_direct_search():
    """Specific queries with qualifiers and price must bypass clarification."""
    thread_id = "test_specific_formal_shoes"
    config = {"configurable": {"thread_id": thread_id}}
    state = {
        "current_query": "Find men's black formal shoes under ₹5,000",
        "original_query": "Find men's black formal shoes under ₹5,000",
        "search_results": [],
        "selected_product": None,
        "upsell_results": [],
        "clarification_count": 0,
    }
    result = agent_app.invoke(state, config=config)
    assert result.get("needs_clarification") is False
    assert result.get("clarification_question") is None

