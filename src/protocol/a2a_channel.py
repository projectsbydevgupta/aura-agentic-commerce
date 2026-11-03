"""
AURA Agent-to-Agent (A2A) Interoperability Channel
Implements Criterion 6 & 7:
Connects heterogeneous agentic frameworks:
Google Agent Development Kit (ADK) Buyer <---> CrewAI Merchant Concierge
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.crews.fashion_crew import get_fashion_crew, CrewConciergeResult

# =====================================================================
# Standard A2A Protocol Envelope Schemas
# =====================================================================

class A2AMessageEnvelope(BaseModel):
    a2a_version: str = "1.0"
    sender_framework: str = Field(..., description="Framework of the sender agent, e.g. Google-ADK")
    sender_agent_id: str
    receiver_framework: str = Field(..., description="Framework of the receiver agent, e.g. CrewAI")
    receiver_agent_id: str
    intent: str  # e.g. "request_outfit_capsule", "request_quote", "execute_purchase"
    payload: Dict[str, Any]
    spending_ceiling_inr: float
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class A2AResponseEnvelope(BaseModel):
    a2a_version: str = "1.0"
    status: str
    responder_framework: str
    responder_agent_id: str
    target_sender_id: str
    payload: Dict[str, Any]
    requires_principal_confirmation: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# =====================================================================
# Interoperability Bridge Engine
# =====================================================================

class A2AInteroperabilityBridge:
    """
    Coordinates heterogeneous Agent-to-Agent transactions between
    Google ADK and CrewAI.
    """

    def __init__(self):
        self.fashion_crew = get_fashion_crew()

    def process_adk_to_crewai_transaction(
        self,
        envelope: A2AMessageEnvelope
    ) -> A2AResponseEnvelope:
        """
        Translates a Google ADK Buyer request, executes the CrewAI Fashion Concierge,
        and returns an A2A standard response envelope.
        """
        payload = envelope.payload
        query = payload.get("query", "luxury wardrobe essentials")
        gender = payload.get("gender")
        budget = envelope.spending_ceiling_inr

        # 1. Execute CrewAI Hierarchical Concierge
        crew_res: CrewConciergeResult = self.fashion_crew.run_hierarchical_crew(
            query=query,
            budget_inr=budget,
            gender=gender
        )

        # 2. Check if total exceeds spending ceiling
        exceeds = crew_res.total_outfit_inr > budget

        # 3. Formulate A2A Response Envelope
        items_payload = [item.model_dump() if hasattr(item, "model_dump") else item.dict() for item in crew_res.outfit_items]
        
        return A2AResponseEnvelope(
            status="SUCCESS" if crew_res.status == "success" else "REJECTED",
            responder_framework="CrewAI",
            responder_agent_id="AURA_Fashion_Supervisor_Crew",
            target_sender_id=envelope.sender_agent_id,
            payload={
                "curated_aesthetic": crew_res.aesthetic_vibe,
                "curated_pieces": items_payload,
                "total_cost_inr": crew_res.total_outfit_inr,
                "supervisor_verdict": crew_res.supervisor_verdict,
                "within_budget_ceiling": not exceeds
            },
            requires_principal_confirmation=exceeds
        )


# Global singleton
_a2a_bridge = A2AInteroperabilityBridge()

def get_a2a_bridge() -> A2AInteroperabilityBridge:
    """Returns global A2AInteroperabilityBridge singleton."""
    return _a2a_bridge

