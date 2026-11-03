"""
Unit & Integration Tests for CrewAI and Google ADK Interoperability.
Validates Criterion 4 (Design Patterns), Criterion 6 (A2A Protocol), and Criterion 7 (CrewAI + ADK ONLY).
"""

import unittest
from src.crews.fashion_crew import get_fashion_crew
from src.adk_agents.buyer_agent import GoogleADKBuyerAgent, ADKBuyerGoal
from src.protocol.a2a_channel import get_a2a_bridge, A2AMessageEnvelope

class TestCrewAIADKInteroperability(unittest.TestCase):

    def setUp(self):
        self.crew = get_fashion_crew()
        self.buyer = GoogleADKBuyerAgent(agent_id="ADK_Test_Buyer_01")
        self.bridge = get_a2a_bridge()

    def test_01_crewai_hierarchical_concierge_execution(self):
        """Tests that CrewAI manager coordinates workers to curate an outfit."""
        res = self.crew.run_hierarchical_crew(
            query="men formal white shirt",
            budget_inr=10000.0,
            gender="Men"
        )
        self.assertEqual(res.status, "success")
        self.assertGreaterEqual(len(res.outfit_items), 1)
        self.assertIsNotNone(res.total_outfit_inr)
        self.assertIn("Luxe", res.aesthetic_vibe)
        self.assertTrue(res.within_budget)

    def test_02_adk_buyer_agent_autonomous_purchase(self):
        """Tests that Google ADK buyer agent autonomously searches, applies voucher, and buys."""
        goal = ADKBuyerGoal(
            query="running shoes",
            max_budget_inr=8000.0,
            shopper_email="patron_aarav@aurafashion.com",
            preferred_gender="Men"
        )
        order = self.buyer.execute_procurement_goal(goal)
        self.assertEqual(order.status, "ORDER_PLACED_A2A")
        self.assertLessEqual(order.net_paid_inr, 8000.0)
        self.assertEqual(order.voucher_applied, "AURA20")
        self.assertGreater(order.discount_savings_inr, 0.0)

    def test_03_adk_buyer_agent_budget_ceiling_defense(self):
        """Tests that Google ADK buyer agent refuses to buy if items exceed the authorized budget limit."""
        # Unrealistic low budget: ₹50 INR (Impossible for luxury shoes)
        goal = ADKBuyerGoal(
            query="luxury sports shoes",
            max_budget_inr=50.0,
            shopper_email="patron_aarav@aurafashion.com"
        )
        order = self.buyer.execute_procurement_goal(goal)
        self.assertEqual(order.status, "BLOCKED_BY_BUDGET_CEILING")
        self.assertIn("Gated Spend Ceiling Violation", order.audit_notes)

    def test_04_heterogeneous_a2a_protocol_envelope_exchange(self):
        """
        Tests end-to-end A2A interoperability:
        Google-ADK Buyer sends A2AMessageEnvelope ---> CrewAI Merchant responds with A2AResponseEnvelope.
        """
        envelope = A2AMessageEnvelope(
            sender_framework="Google-ADK",
            sender_agent_id="ADK_Autonomous_Buyer_007",
            receiver_framework="CrewAI",
            receiver_agent_id="AURA_Merchant_Concierge_Crew",
            intent="request_outfit_capsule",
            payload={"query": "black running sneakers", "gender": "Men"},
            spending_ceiling_inr=6000.0
        )

        response = self.bridge.process_adk_to_crewai_transaction(envelope)
        
        self.assertEqual(response.a2a_version, "1.0")
        self.assertEqual(response.responder_framework, "CrewAI")
        self.assertEqual(response.target_sender_id, "ADK_Autonomous_Buyer_007")
        self.assertEqual(response.status, "SUCCESS")
        self.assertIn("curated_pieces", response.payload)
        self.assertIn("total_cost_inr", response.payload)

    def test_05_adk_envelope_creation_and_remote_dispatch(self):
        """Tests that GoogleADKBuyerAgent creates envelopes and dispatches to merchant crew."""
        env = self.buyer.create_a2a_envelope(
            query="luxury silk tie",
            spending_ceiling_inr=5000.0,
            gender="Men"
        )
        self.assertEqual(env.sender_framework, "Google-ADK")
        self.assertEqual(env.receiver_framework, "CrewAI")
        self.assertEqual(env.spending_ceiling_inr, 5000.0)

        # Dispatch via remote call (with local fallback if test server offline)
        res = self.buyer.call_remote_merchant_crew(env)
        self.assertIn(res.status, ["SUCCESS", "REJECTED"])
        self.assertEqual(res.responder_framework, "CrewAI")

    def test_06_http_crew_concierge_endpoint(self):
        """Tests the FastAPI HTTP endpoint POST /protocol/v1/crew/concierge."""
        import asyncio
        import httpx
        from server import app

        async def _test_endpoint():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
                payload = {
                    "a2a_version": "1.0",
                    "sender_framework": "Google-ADK",
                    "sender_agent_id": "ADK_Test_Runner",
                    "receiver_framework": "CrewAI",
                    "receiver_agent_id": "AURA_Fashion_Supervisor_Crew",
                    "intent": "request_outfit_capsule",
                    "payload": {"query": "white cotton shirt", "gender": "Men"},
                    "spending_ceiling_inr": 8000.0
                }
                resp = await client.post("/protocol/v1/crew/concierge", json=payload)
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertEqual(data["responder_framework"], "CrewAI")
                self.assertIn("curated_pieces", data["payload"])

        asyncio.run(_test_endpoint())

if __name__ == "__main__":
    unittest.main()
