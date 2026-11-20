"""
Unit Tests for Arize Phoenix OpenTelemetry Tracing Engine
Criterion 9: Observability, Tracing, Auditability & Provenance
"""

import unittest
from src.monitoring.phoenix_tracer import get_phoenix_tracer, PhoenixTracer
from src.monitoring.callbacks import AgentExecutionMonitorCallback

class TestPhoenixTracing(unittest.TestCase):

    def setUp(self):
        self.tracer = get_phoenix_tracer()

    def test_01_tracer_initialization(self):
        status = self.tracer.get_status()
        self.assertTrue(status["enabled"])
        self.assertTrue(status["opentelemetry_installed"])
        self.assertEqual(status["project_name"], "aura-commerce")
        self.assertEqual(status["local_port"], 6006)
        self.assertEqual(status["cloud_endpoint"], "https://app.phoenix.arize.com/s/aura-commerce/v1/traces")

    def test_02_span_lifecycle(self):
        step_id = "test_step_px_001"
        thread_id = "test_thread_px_001"

        # Root pipeline span
        root_span = self.tracer.start_agent_span(
            step_id=f"root_{step_id}",
            thread_id=thread_id,
            agent_name="Conversational Pipeline",
            input_data="Need an evening gown under 5000"
        )
        self.assertIsNotNone(root_span)

        # Child agent span
        child_span = self.tracer.start_agent_span(
            step_id=step_id,
            thread_id=thread_id,
            agent_name="Query Agent",
            input_data="Need an evening gown under 5000"
        )
        self.assertIsNotNone(child_span)

        # End child span
        self.tracer.end_agent_span(
            step_id=step_id,
            output_data="Intent: search, Category: gowns",
            prompt_tokens=85,
            completion_tokens=20
        )

        # End root span
        self.tracer.end_agent_span(
            step_id=f"root_{step_id}",
            output_data="Returned 4 evening gowns",
            prompt_tokens=85,
            completion_tokens=20
        )

        status = self.tracer.get_status()
        self.assertGreaterEqual(status["total_traces_emitted"], 1)

    def test_03_monitor_callback_integration(self):
        monitor = AgentExecutionMonitorCallback()
        step_id = "callback_trace_step_002"
        thread_id = "thread_callback_002"

        # Calling on_agent_start should trigger phoenix span
        step = monitor.on_agent_start(step_id, thread_id, "Validation Agent", "Product candidate list")
        self.assertEqual(step.agent_name, "Validation Agent")

        # Calling on_agent_end should complete phoenix span
        completed = monitor.on_agent_end(step_id, "All products validated", prompt_tokens=50, completion_tokens=15)
        self.assertEqual(completed.status, "completed")
        self.assertEqual(completed.total_tokens, 65)


if __name__ == "__main__":
    unittest.main()

