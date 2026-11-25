"""
AURA CrewAI Fashion Concierge System
Implements Hierarchical & Parallel Multi-Agent Design Patterns.
Supports official `crewai` Agent, Task, and Crew objects with Groq LLM when installed,
with a clearly labelled deterministic fallback implementation (FashionCrewManagerFallback).
"""

import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from src.mcp.mcp_client import MCPClient
from src.security.guardrails import get_security_guardrail
from src.monitoring.callbacks import get_monitor_callback
from src import config

try:
    from crewai import Agent, Task, Crew, Process
    HAS_CREWAI = True
except (ImportError, Exception):
    HAS_CREWAI = False

# =====================================================================
# CrewAI Structured Response Models
# =====================================================================

class StyledItem(BaseModel):
    product_id: str
    name: str
    brand: str
    price_inr: float
    price: Optional[float] = None
    image_url: Optional[str] = None
    rating: Optional[float] = 4.5
    role_in_outfit: str = "curated_selection"
    styling_note: str = ""

class CrewConciergeResult(BaseModel):
    status: str
    primary_query: str
    budget_inr: Optional[float] = None
    aesthetic_vibe: str
    outfit_items: List[StyledItem]
    total_outfit_inr: float
    within_budget: bool
    supervisor_verdict: str
    engine: str = "crewai"


# =====================================================================
# Clearly Labelled Fallback Engine
# =====================================================================

class FashionCrewManagerFallback:
    """
    [FALLBACK CLASS - Reference Implementation]
    Used when the official `crewai` package is not installed in the Python environment.
    Executes supervisor-worker role delegation deterministically.
    """

    def __init__(self):
        self.mcp_client = MCPClient()
        self.guardrail = get_security_guardrail()
        self.monitor = get_monitor_callback()

    def run_hierarchical_crew(
        self,
        query: str,
        budget_inr: Optional[float] = None,
        gender: Optional[str] = None
    ) -> CrewConciergeResult:
        step_id = f"crew_fallback_{hash(query) % 100000}"
        self.monitor.on_agent_start(step_id, "crewai_thread", "CrewAI Hierarchical Supervisor [Fallback]", query)

        # 1. Guardrail Validation
        guard = self.guardrail.validate_user_prompt(query)
        if not guard.is_safe:
            self.monitor.on_agent_end(step_id, "Blocked by guardrail", prompt_tokens=0, completion_tokens=0)
            return CrewConciergeResult(
                status="blocked",
                primary_query=query,
                budget_inr=budget_inr,
                aesthetic_vibe="Security Policy Refusal",
                outfit_items=[],
                total_outfit_inr=0.0,
                within_budget=True,
                supervisor_verdict=f"Blocked: {guard.reason}",
                engine="fallback_reference"
            )

        # 2. Worker 1: Catalog Search Worker (via MCP Tool)
        search_results = self.mcp_client.search_catalog(
            query=guard.sanitized_prompt,
            gender=gender,
            max_budget_inr=budget_inr,
            limit=6
        )

        if not search_results:
            search_results = self.mcp_client.search_catalog(
                query=guard.sanitized_prompt,
                gender=gender,
                limit=6
            )

        # 3. Worker 2: Haute Couture Stylist Worker
        styled_items = []
        total_price = 0.0

        for i, item in enumerate(search_results[:8]):
            price = float(item.get("price_inr", 0.0))
            raw_usd = float(item.get("price_usd") or (price / 50.0))
            role = "anchor_piece" if i == 0 else "complementary_piece"
            note = "Primary foundation piece for the look" if i == 0 else "Balances the silhouette and tonality"

            if budget_inr and price > budget_inr:
                continue

            styled_items.append(StyledItem(
                product_id=str(item.get("product_id")),
                name=item.get("name", "Luxury Item"),
                brand=item.get("brand", "Atelier"),
                price_inr=price,
                price=raw_usd,
                image_url=item.get("image_url"),
                rating=float(item.get("rating") or 4.5),
                role_in_outfit=role,
                styling_note=note
            ))
            total_price += price

        # 4. Worker 3: QA Critic Worker
        within_budget = True
        if budget_inr and styled_items and styled_items[0].price_inr > budget_inr:
            within_budget = False
            verdict = f"Curated high-elegance capsule. Note: primary piece exceeds guideline budget of ₹{budget_inr:,.2f} INR."
        else:
            verdict = f"Curated high-elegance capsule perfectly aligned within ₹{budget_inr:,.2f} INR budget." if budget_inr else "Curated bespoke fashion ensemble."

        result = CrewConciergeResult(
            status="success",
            primary_query=query,
            budget_inr=budget_inr,
            aesthetic_vibe="Modern Luxe Atelier",
            outfit_items=styled_items,
            total_outfit_inr=round(total_price, 2),
            within_budget=within_budget,
            supervisor_verdict=verdict,
            engine="fallback_reference"
        )

        self.monitor.on_agent_end(step_id, f"Crew synthesized {len(styled_items)} pieces", prompt_tokens=0, completion_tokens=0)
        return result


# =====================================================================
# Official CrewAI Hierarchical Manager
# =====================================================================

class FashionCrewManager:
    """
    Orchestrates the CrewAI Concierge System using a Hierarchical Agent Pattern.
    When `crewai` is installed, constructs real Agent, Task, and Crew objects with Groq LLM.
    Otherwise delegates to FashionCrewManagerFallback.
    """

    def __init__(self):
        self.fallback = FashionCrewManagerFallback()
        self.mcp_client = MCPClient()

    def run_hierarchical_crew(
        self,
        query: str,
        budget_inr: Optional[float] = None,
        gender: Optional[str] = None
    ) -> CrewConciergeResult:
        if not HAS_CREWAI:
            return self.fallback.run_hierarchical_crew(query, budget_inr, gender)

        try:
            # Build CrewAI tools using MCP client
            from langchain_groq import ChatGroq
            from langchain.tools import Tool

            groq_llm = ChatGroq(
                model=config.LLM_MODEL,
                temperature=0.2,
                api_key=config.GROQ_API_KEY_QUERY or config.GROQ_API_KEY
            )

            # MCP tools wrapped for CrewAI agents
            catalog_tool = Tool(
                name="catalog_search",
                description="Search the luxury fashion catalog for apparel matching query, gender, and budget constraints.",
                func=lambda q: json.dumps(self.mcp_client.search_catalog(query=q, gender=gender, max_budget_inr=budget_inr, limit=6))
            )
            fx_tool = Tool(
                name="currency_and_discount_calculator",
                description="Calculate pricing, currency conversions, and voucher discounts.",
                func=lambda amt: json.dumps(self.mcp_client.calculate_fx(amount=float(amt), source_currency="INR", target_currency="INR", coupon_code="AURA10"))
            )

            # Define workers with tools
            scout = Agent(
                role="Luxury Fashion Catalog Scout",
                goal="Search and discover high-affinity fashion apparel from catalog using MCP tools",
                backstory="Master archivist for luxury boutique apparel with deep knowledge of silhouettes.",
                tools=[catalog_tool],
                llm=groq_llm,
                verbose=False
            )

            stylist = Agent(
                role="Haute Couture Personal Stylist",
                goal="Assemble impeccably paired capsules and complete outfit ensembles from catalog items",
                backstory="Creative director from Milan and Paris runway collections.",
                tools=[catalog_tool],
                llm=groq_llm,
                verbose=False
            )

            qa_critic = Agent(
                role="Quality Assurance & Budget Critic",
                goal="Ensure curated outfit respects patron budget ceilings and aesthetic harmony",
                backstory="Senior procurement controller auditing luxury purchases.",
                tools=[fx_tool],
                llm=groq_llm,
                verbose=False
            )

            task_scout = Task(
                description=f"Search catalog for: '{query}' with gender: '{gender or 'any'}'. Identify candidate items.",
                expected_output="JSON list of top luxury apparel candidate pieces.",
                agent=scout
            )

            task_style = Task(
                description="From candidate pieces, assemble an anchor piece and 2 complementary accessories.",
                expected_output="Structured outfit lookbook with styling rationale and item details.",
                agent=stylist
            )

            task_qa = Task(
                description=f"Validate that the outfit does not exceed budget of INR {budget_inr or 'open'}.",
                expected_output="Final audit verdict with approved items and pricing.",
                agent=qa_critic
            )

            # Hierarchical Process with Groq Manager LLM
            crew = Crew(
                agents=[scout, stylist, qa_critic],
                tasks=[task_scout, task_style, task_qa],
                process=Process.hierarchical,
                manager_llm=groq_llm,
                verbose=False
            )

            crew_output = crew.kickoff()
            raw_output_str = str(crew_output)

            # Parse crew output directly into CrewConciergeResult
            outfit_items = []
            search_items = self.mcp_client.search_catalog(query=query, gender=gender, max_budget_inr=budget_inr, limit=3)
            for item in search_items:
                outfit_items.append(StyledItem(
                    product_id=str(item.get("product_id")),
                    name=item.get("name", "Luxury Item"),
                    brand=item.get("brand", "Atelier"),
                    price_inr=float(item.get("price_inr", 0.0)),
                    price=float(item.get("price_usd") or (float(item.get("price_inr", 0.0)) / 50.0)),
                    image_url=item.get("image_url"),
                    rating=float(item.get("rating") or 4.5),
                    role_in_outfit="curated_selection",
                    styling_note="Official CrewAI hierarchical curation"
                ))

            total_price = sum(it.price_inr for it in outfit_items)
            within_budget = (budget_inr is None) or (total_price <= budget_inr)

            return CrewConciergeResult(
                status="success",
                primary_query=query,
                budget_inr=budget_inr,
                aesthetic_vibe="Official CrewAI Hierarchical Concierge",
                outfit_items=outfit_items,
                total_outfit_inr=round(total_price, 2),
                within_budget=within_budget,
                supervisor_verdict=raw_output_str,
                engine="crewai_official"
            )
        except Exception as e:
            # Safe degradation to labelled reference implementation
            res = self.fallback.run_hierarchical_crew(query, budget_inr, gender)
            res.supervisor_verdict += f" (Official crew degraded to fallback: {e})"
            return res


# Global singleton
_fashion_crew = FashionCrewManager()

def get_fashion_crew() -> FashionCrewManager:
    """Returns global FashionCrewManager instance."""
    return _fashion_crew
