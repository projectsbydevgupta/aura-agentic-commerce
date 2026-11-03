"""
Unit & Integration Tests for Blackboard Architecture & Pub-Sub Messaging.
Validates Criterion 1 (Mandatory Blackboard Context Sharing & Messaging Architecture comparison).
"""

import unittest
from src.blackboard.blackboard import get_blackboard_arbiter
from src.blackboard.event_bus import EventBus, run_messaging_architecture_comparison

class TestBlackboardAndMessaging(unittest.TestCase):

    def setUp(self):
        self.arbiter = get_blackboard_arbiter()
        self.bus = EventBus()

    def test_01_blackboard_solves_fashion_query(self):
        """Tests that the Blackboard Controller loop coordinates Knowledge Sources to completion."""
        state = self.arbiter.solve(
            session_id="session_test_01",
            query="men formal white shirt",
            budget_inr=5000.0,
            gender="Men"
        )
        self.assertTrue(state.is_solved)
        self.assertEqual(state.intent, "search")
        self.assertEqual(state.validation_status, "VALIDATED")
        self.assertGreater(len(state.curated_ensemble), 0)
        self.assertGreater(len(state.event_trail), 3)

    def test_02_blackboard_trips_clarification_gate_on_vague_input(self):
        """Tests that Context Knowledge Source pauses execution when query is ambiguous."""
        state = self.arbiter.solve(
            session_id="session_vague_01",
            query="clothes"
        )
        self.assertTrue(state.clarification_needed)
        self.assertIsNotNone(state.clarification_question)
        self.assertEqual(len(state.candidate_products), 0)

    def test_03_pubsub_event_bus_delivery(self):
        """Tests publish-subscribe message routing across topics."""
        received = []
        def handler(msg):
            received.append(msg.payload.get("status"))

        self.bus.subscribe("cart.abandoned", handler)
        self.bus.publish("cart.abandoned", "CartDetector", {"status": "needs_recovery"})

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0], "needs_recovery")

    def test_04_messaging_architecture_comparative_benchmark(self):
        """Tests side-by-side execution comparison between Blackboard and Pub-Sub."""
        report = run_messaging_architecture_comparison(
            query="casual blue sneakers",
            budget_inr=3000.0
        )
        self.assertIn("blackboard", report)
        self.assertIn("pubsub", report)
        self.assertIn("latency_ms", report["blackboard"])
        self.assertIn("latency_ms", report["pubsub"])
        self.assertIn("verdict", report)

if __name__ == "__main__":
    unittest.main()

