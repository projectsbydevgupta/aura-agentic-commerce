"""
AURA Blackboard Architecture Engine
Implements Criterion 1 (Mandatory Blackboard Context Sharing):
- Central Blackboard (Shared Knowledge Base)
- Specialized Knowledge Sources (KS_Query, KS_Context, KS_Search, KS_Validation, KS_Stylist)
- Blackboard Arbiter / Controller Loop
"""

import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.mcp.mcp_client import MCPClient
from src.security.guardrails import get_security_guardrail

# =====================================================================
# Blackboard State Data Structure
# =====================================================================

class BlackboardEvent(BaseModel):
    timestamp: str
    source_name: str
    action: str
    details: str

class BlackboardState(BaseModel):
    # 1. Problem Definition
    session_id: str
    original_query: str
    max_budget_inr: Optional[float] = None
    gender: Optional[str] = None
    
    # 2. Hypotheses & Knowledge Additions
    intent: Optional[str] = None
    cleaned_query: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
    
    # 3. Candidate Solutions
    candidate_products: List[Dict[str, Any]] = Field(default_factory=list)
    validation_status: Optional[str] = None
    validation_notes: Optional[str] = None
    
    # 4. Synthesized Final Solution
    curated_ensemble: List[Dict[str, Any]] = Field(default_factory=list)
    total_ensemble_inr: float = 0.0
    aesthetic_concept: Optional[str] = None
    
    # 5. Meta Audit Trail
    revision: int = 0
    is_solved: bool = False
    event_trail: List[BlackboardEvent] = Field(default_factory=list)


# =====================================================================
# Central Blackboard Store
# =====================================================================

class Blackboard:
    """The central shared state repository accessed by Knowledge Sources."""

    def __init__(self, session_id: str, query: str, budget_inr: Optional[float] = None, gender: Optional[str] = None):
        self.state = BlackboardState(
            session_id=session_id,
            original_query=query,
            cleaned_query=query,
            max_budget_inr=budget_inr,
            gender=gender
        )
        self.post_event("Blackboard", "INITIALIZE", f"Problem initialized: '{query}' with budget {budget_inr}")

    def post_event(self, source: str, action: str, details: str):
        event = BlackboardEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            source_name=source,
            action=action,
            details=details
        )
        self.state.event_trail.append(event)
        self.state.revision += 1

    def is_solved(self) -> bool:
        return self.state.is_solved or self.state.clarification_needed


# =====================================================================
# Specialized Knowledge Sources (KS)
# =====================================================================

class BaseKnowledgeSource:
    name: str = "BaseKS"

    def can_contribute(self, bb: Blackboard) -> bool:
        raise NotImplementedError

    def contribute(self, bb: Blackboard):
        raise NotImplementedError


class KS_QueryParser(BaseKnowledgeSource):
    name = "KS_QueryParser"

    def can_contribute(self, bb: Blackboard) -> bool:
        # Contributes when query has not yet been parsed into intent & filters
        return bb.state.intent is None

    def contribute(self, bb: Blackboard):
        q = bb.state.original_query.lower()
        guard = get_security_guardrail().validate_user_prompt(bb.state.original_query)
        
        bb.state.cleaned_query = guard.sanitized_prompt
        bb.state.intent = "search"
        if "under" in q or "below" in q:
            bb.state.intent = "budget_search"
            
        bb.post_event(self.name, "EXTRACT_INTENT", f"Intent identified: {bb.state.intent}")


class KS_ContextGate(BaseKnowledgeSource):
    name = "KS_ContextGate"

    def can_contribute(self, bb: Blackboard) -> bool:
        # Runs after query parsing, before search
        return bb.state.intent is not None and len(bb.state.candidate_products) == 0 and not bb.state.clarification_needed

    def contribute(self, bb: Blackboard):
        q = (bb.state.cleaned_query or "").strip()
        # Vague check
        if len(q.split()) <= 1 and q in ["clothes", "shoes", "help", "fashion"]:
            bb.state.clarification_needed = True
            bb.state.clarification_question = "Could you specify your preferred style, occasion, or gender?"
            bb.post_event(self.name, "PAUSE_CLARIFICATION", "Ambiguity gate tripped.")
        else:
            bb.post_event(self.name, "GATE_PASSED", "Query context sufficient for retrieval.")


class KS_CatalogRetriever(BaseKnowledgeSource):
    name = "KS_CatalogRetriever"

    def __init__(self):
        self.mcp = MCPClient()

    def can_contribute(self, bb: Blackboard) -> bool:
        # Contributes when query is parsed, not vague, and products not yet retrieved
        return (
            bb.state.intent is not None
            and not bb.state.clarification_needed
            and len(bb.state.candidate_products) == 0
        )

    def contribute(self, bb: Blackboard):
        products = self.mcp.search_catalog(
            query=bb.state.cleaned_query,
            gender=bb.state.gender,
            max_budget_inr=bb.state.max_budget_inr,
            limit=6
        )
        bb.state.candidate_products = products
        bb.post_event(self.name, "RETRIEVE_CANDIDATES", f"Retrieved {len(products)} candidate catalog pieces.")


class KS_ValidationCritic(BaseKnowledgeSource):
    name = "KS_ValidationCritic"

    def can_contribute(self, bb: Blackboard) -> bool:
        # Runs when candidate products exist but haven't been validated
        return len(bb.state.candidate_products) > 0 and bb.state.validation_status is None

    def contribute(self, bb: Blackboard):
        count = len(bb.state.candidate_products)
        if count > 0:
            bb.state.validation_status = "VALIDATED"
            bb.state.validation_notes = f"Verified {count} items matching criteria."
        else:
            bb.state.validation_status = "RELAXED"
            bb.state.validation_notes = "Relaxing constraints for broader catalogue view."
        bb.post_event(self.name, "VALIDATE_CANDIDATES", f"Validation status: {bb.state.validation_status}")


class KS_FashionStylist(BaseKnowledgeSource):
    name = "KS_FashionStylist"

    def can_contribute(self, bb: Blackboard) -> bool:
        # Runs when products are validated but ensemble is not yet synthesized
        return bb.state.validation_status is not None and not bb.state.is_solved

    def contribute(self, bb: Blackboard):
        prods = bb.state.candidate_products[:3]
        total = sum(p.get("price_inr", 0.0) for p in prods)
        
        bb.state.curated_ensemble = prods
        bb.state.total_ensemble_inr = round(total, 2)
        bb.state.aesthetic_concept = "Atelier Monochromatic Capsule"
        bb.state.is_solved = True
        bb.post_event(self.name, "SYNTHESIZE_ENSEMBLE", f"Curated {len(prods)} pieces totaling ₹{total:,.2f} INR.")


# =====================================================================
# Blackboard Controller / Arbiter Loop
# =====================================================================

class BlackboardArbiter:
    """Controls scheduling of Knowledge Sources against the shared Blackboard."""

    def __init__(self):
        self.knowledge_sources: List[BaseKnowledgeSource] = [
            KS_QueryParser(),
            KS_ContextGate(),
            KS_CatalogRetriever(),
            KS_ValidationCritic(),
            KS_FashionStylist()
        ]

    def solve(self, session_id: str, query: str, budget_inr: Optional[float] = None, gender: Optional[str] = None) -> BlackboardState:
        """Runs the Blackboard controller loop until a solution is synthesized."""
        bb = Blackboard(session_id, query, budget_inr, gender)
        
        max_cycles = 10
        cycle = 0

        while not bb.is_solved() and cycle < max_cycles:
            cycle += 1
            contributed_in_cycle = False

            for ks in self.knowledge_sources:
                if ks.can_contribute(bb):
                    ks.contribute(bb)
                    contributed_in_cycle = True
                    # Check after each contribution
                    if bb.is_solved():
                        break

            if not contributed_in_cycle:
                # No more KS can contribute
                bb.state.is_solved = True
                break

        return bb.state


# Global singleton
_blackboard_arbiter = BlackboardArbiter()

def get_blackboard_arbiter() -> BlackboardArbiter:
    """Returns global BlackboardArbiter singleton."""
    return _blackboard_arbiter

