from typing import Dict, Any, Literal, Optional

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    StateGraph, END, MemorySaver = None, None, None
    LANGGRAPH_AVAILABLE = False

from src.agents.state import AgentState
from src.agents.nodes import (
    query_agent_node,
    context_agent_node,
    search_node,
    validation_agent_node,
    upsell_agent_node
)

# ==========================================
# Conditional Routing Functions
# ==========================================

def route_after_context(state: AgentState) -> Literal["search_node", "__end__"]:
    """
    Decides whether to proceed to search or pause to ask clarification from user.
    """
    if state.get("needs_clarification", False):
        return "__end__"
    return "search_node"


def route_after_validation(state: AgentState) -> Literal["upsell_agent", "search_node", "__end__"]:
    """
    Decides whether validation passed (proceed to upsell/complete)
    or failed and needs a search retry, or exhausted retries without matches.
    """
    val_res = state.get("validation_result", {})
    is_valid = val_res.get("validated", True)
    retry_count = val_res.get("retry_count", 0)

    if is_valid:
        return "upsell_agent"
    if retry_count >= 2:
        return "__end__"
    return "search_node"


# ==========================================
# Build LangGraph StateGraph with MemorySaver Checkpointer
# ==========================================

memory_checkpointer = MemorySaver() if LANGGRAPH_AVAILABLE else {}

def create_ecommerce_agent_graph(checkpointer: Optional[Any] = memory_checkpointer):
    """Assembles and compiles the multi-agent LangGraph workflow if installed."""
    if not LANGGRAPH_AVAILABLE:
        return None
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("query_agent", query_agent_node)
    workflow.add_node("context_agent", context_agent_node)
    workflow.add_node("search_node", search_node)
    workflow.add_node("validation_agent", validation_agent_node)
    workflow.add_node("upsell_agent", upsell_agent_node)

    # 2. Set Entry Point
    workflow.set_entry_point("query_agent")

    # 3. Add Edges
    workflow.add_edge("query_agent", "context_agent")

    # Conditional Branch after Context Agent:
    workflow.add_conditional_edges(
        "context_agent",
        route_after_context,
        {
            "search_node": "search_node",
            "__end__": END
        }
    )

    workflow.add_edge("search_node", "validation_agent")

    # Conditional Branch after Validation Agent:
    workflow.add_conditional_edges(
        "validation_agent",
        route_after_validation,
        {
            "upsell_agent": "upsell_agent",
            "search_node": "search_node",
            "__end__": END
        }
    )

    workflow.add_edge("upsell_agent", END)

    return workflow.compile(checkpointer=checkpointer)


# ==========================================
# Pure Python Sequential Pipeline Fallback
# ==========================================

class SequentialEcommerceWorkflow:
    """
    Native sequential multi-agent pipeline orchestrator.
    Mirrors StateGraph execution pattern identically:
    query_agent -> context_agent -> (conditional: clarification?) -> search_node -> validation_agent -> (conditional: retry?) -> upsell_agent
    """
    def __init__(self, checkpointer: Optional[Dict[str, Any]] = None):
        self.checkpointer = checkpointer if checkpointer is not None else {}

    def invoke(self, input_state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        thread_id = None
        if config and "configurable" in config:
            thread_id = config["configurable"].get("thread_id")

        state: AgentState = dict(input_state)
        if thread_id and thread_id in self.checkpointer:
            saved = self.checkpointer[thread_id]
            for k, v in saved.items():
                if k not in state:
                    state[k] = v

        # 1. Query Agent Node
        res1 = query_agent_node(state)
        state.update(res1)

        # 2. Context Agent Node
        res2 = context_agent_node(state)
        state.update(res2)

        # Conditional branch after context
        if route_after_context(state) == "__end__":
            state["search_results"] = []
            state["selected_product"] = None
            state["upsell_results"] = []
            if thread_id:
                self.checkpointer[thread_id] = state
            return state

        # Search + Validation Loop (Max 2 retries)
        max_retries = 2
        for _ in range(max_retries + 1):
            # 3. Search Node (Hybrid Search with Price Intelligence)
            res3 = search_node(state)
            state.update(res3)

            # 4. Validation Agent Node
            res4 = validation_agent_node(state)
            state.update(res4)

            # Conditional branch after validation
            branch = route_after_validation(state)
            if branch == "upsell_agent":
                break
            elif branch == "__end__":
                state["search_results"] = []
                state["selected_product"] = None
                state["upsell_results"] = []
                break

        # 5. Upsell Agent Node (only if validated with real catalog pieces)
        if state.get("validation_result", {}).get("validated") is True and state.get("search_results"):
            if not state.get("selected_product"):
                state["selected_product"] = state["search_results"][0]
            res5 = upsell_agent_node(state)
            state.update(res5)
        else:
            state["selected_product"] = None
            state["upsell_results"] = []

        if thread_id:
            self.checkpointer[thread_id] = state

        return state


# Global compiled app with conversation checkpointing
if LANGGRAPH_AVAILABLE:
    agent_app = create_ecommerce_agent_graph()
else:
    agent_app = SequentialEcommerceWorkflow(memory_checkpointer)
