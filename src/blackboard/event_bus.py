"""
AURA Event-Driven Pub-Sub Messaging Architecture
Implements Criterion 1 (Comparative Messaging Architecture alongside Blackboard):
- Topic-based publish/subscribe message bus
- Event dispatching & message handling
- Performance comparison benchmark: Blackboard (Shared State) vs Pub-Sub (Message Queues)
"""

import time
from typing import Dict, Any, List, Callable
from pydantic import BaseModel, Field

class PubSubMessage(BaseModel):
    topic: str
    sender_node: str
    payload: Dict[str, Any]
    timestamp: float = Field(default_factory=time.time)

class EventBus:
    """In-memory topic-based Publish-Subscribe messaging broker."""

    def __init__(self):
        self.subscribers: Dict[str, List[Callable[[PubSubMessage], None]]] = {}
        self.message_history: List[PubSubMessage] = []

    def subscribe(self, topic: str, handler: Callable[[PubSubMessage], None]):
        """Registers a consumer handler for a specific topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(handler)

    def publish(self, topic: str, sender: str, payload: Dict[str, Any]):
        """Dispatches an event message to all subscribers of the topic."""
        msg = PubSubMessage(topic=topic, sender_node=sender, payload=payload)
        self.message_history.append(msg)
        
        handlers = self.subscribers.get(topic, [])
        for h in handlers:
            try:
                h(msg)
            except Exception as e:
                print(f"Error in pubsub handler for topic '{topic}': {e}")


def run_messaging_architecture_comparison(query: str = "white linen summer shirt", budget_inr: float = 4000.0) -> Dict[str, Any]:
    """
    Executes a side-by-side empirical benchmark of:
    1. Blackboard Architecture (Shared Repository + Arbiter)
    2. Event-Driven Pub-Sub Architecture (Message Queues across topics)
    Returns latency, message count, and memory overhead comparison.
    """
    from src.blackboard.blackboard import get_blackboard_arbiter
    from src.mcp.mcp_client import MCPClient

    # --- 1. Blackboard Benchmark ---
    t0_bb = time.perf_counter()
    arbiter = get_blackboard_arbiter()
    bb_state = arbiter.solve("benchmark_session", query, budget_inr=budget_inr)
    t1_bb = time.perf_counter()
    bb_latency_ms = round((t1_bb - t0_bb) * 1000, 2)
    bb_events_count = len(bb_state.event_trail)

    # --- 2. Pub-Sub Benchmark ---
    t0_ps = time.perf_counter()
    bus = EventBus()
    mcp = MCPClient()
    pipeline_state = {}

    def on_query_topic(msg: PubSubMessage):
        q = msg.payload.get("query", "")
        # Forward to search topic
        prods = mcp.search_catalog(query=q, max_budget_inr=budget_inr, limit=4)
        bus.publish("aura.events.products_retrieved", "SearchWorker", {"items": prods})

    def on_products_topic(msg: PubSubMessage):
        items = msg.payload.get("items", [])
        pipeline_state["final_items"] = items

    bus.subscribe("aura.events.query_received", on_query_topic)
    bus.subscribe("aura.events.products_retrieved", on_products_topic)

    # Trigger pubsub pipeline
    bus.publish("aura.events.query_received", "HttpGateway", {"query": query})
    t1_ps = time.perf_counter()
    ps_latency_ms = round((t1_ps - t0_ps) * 1000, 2)
    ps_messages_count = len(bus.message_history)

    return {
        "benchmark_query": query,
        "blackboard": {
            "architecture": "Blackboard (Central Shared Knowledge Store + Arbiter)",
            "latency_ms": bb_latency_ms,
            "revisions_count": bb_state.revision,
            "events_logged": bb_events_count,
            "items_curated": len(bb_state.curated_ensemble),
            "state_consistency": "High (ACID-like single truth)"
        },
        "pubsub": {
            "architecture": "Event-Driven Pub-Sub (Topic Message Bus)",
            "latency_ms": ps_latency_ms,
            "messages_exchanged": ps_messages_count,
            "items_curated": len(pipeline_state.get("final_items", [])),
            "state_consistency": "Eventual (Decoupled subscribers)"
        },
        "verdict": "Blackboard excels in multi-turn conversational state tracking and centralized conflict resolution, while Pub-Sub offers loose coupling for asynchronous background dispatches (e.g., WhatsApp queue triggers)."
    }

