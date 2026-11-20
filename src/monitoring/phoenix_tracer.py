"""
AURA Arize Phoenix OpenTelemetry Tracing Engine
Criterion 9: Observability, Tracing, Auditability & Provenance.

Instruments the Multi-Agent System (MAS) pipeline:
- Root Trace: Conversational Pipeline (LangGraph execution flow)
- Agent Spans: Query Agent, Context Agent, Validation Agent, Upsell Agent, Campaign Agent
- Tool Spans: Search Node (MongoDB Atlas Hybrid Vector Search), MCP tools
- Token Metrics: Prompt tokens, completion tokens, latency, input/output values

Exports traces via OpenTelemetry HTTP/protobuf to:
1. Arize Phoenix Cloud: https://app.phoenix.arize.com/s/<space-name>/v1/traces
2. Local Phoenix Port: http://127.0.0.1:<port>/v1/traces (default 6006)
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("aura.phoenix")

# Ensure proper CA bundle for SSL verification on Windows / MinGW
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except Exception:
    pass

# OpenTelemetry imports
try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode, Status
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning("OpenTelemetry SDK not installed; Phoenix tracing will run in mock mode.")


class PhoenixTracer:
    """
    Centralized Arize Phoenix OpenTelemetry tracing provider.
    Maintains active traces across multi-agent turns with parent-child span hierarchy.
    """

    def __init__(self):
        self.enabled = os.getenv("OBSERVABILITY_TRACING", "true").lower() in ("true", "1", "yes")
        self.endpoint = os.getenv(
            "PHOENIX_COLLECTOR_ENDPOINT",
            "https://app.phoenix.arize.com/s/aura-commerce/v1/traces"
        )
        self.api_key = os.getenv(
            "PHOENIX_API_KEY",
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJqdGkiOiJBcGlLZXk6MSJ9.As3Yw3IgHqcJMtFM33_EhtPHZhakX-bY6QENPovwMrQ"
        )
        self.project_name = os.getenv("PHOENIX_PROJECT_NAME", "aura-commerce")
        self.local_port = int(os.getenv("PHOENIX_LOCAL_PORT", "6006"))

        self.tracer = None
        self.provider = None
        self.active_spans: Dict[str, Any] = {}
        self.root_contexts: Dict[str, Any] = {}
        self.trace_counter: int = 0
        self.last_trace_time: Optional[float] = None

        if OPENTELEMETRY_AVAILABLE and self.enabled:
            self._init_tracer()

    def _init_tracer(self):
        """Initializes OpenTelemetry TracerProvider with Cloud and Local OTLP exporters."""
        try:
            resource = Resource.create({
                "service.name": "aura-agentic-commerce",
                "openinference.project.name": self.project_name,
                "model_id": self.project_name,
                "telemetry.sdk.language": "python",
                "framework": "langgraph-mas"
            })
            self.provider = TracerProvider(resource=resource)

            # 1. Cloud Exporter (Arize Phoenix Cloud space)
            if self.endpoint:
                headers = {}
                if self.api_key:
                    headers["api_key"] = self.api_key
                    headers["Authorization"] = f"Bearer {self.api_key}"
                try:
                    cloud_exporter = OTLPSpanExporter(
                        endpoint=self.endpoint,
                        headers=headers,
                        timeout=3
                    )
                    self.provider.add_span_processor(BatchSpanProcessor(
                        cloud_exporter,
                        max_queue_size=256,
                        schedule_delay_millis=5000,
                        export_timeout_millis=1500
                    ))
                except Exception as ce:
                    logger.debug(f"Cloud Phoenix exporter init skipped: {ce}")

            # 2. Local Port Exporter (for localhost:6006 local viewer)
            if self.local_port:
                local_endpoint = f"http://127.0.0.1:{self.local_port}/v1/traces"
                try:
                    local_exporter = OTLPSpanExporter(
                        endpoint=local_endpoint,
                        timeout=2
                    )
                    self.provider.add_span_processor(BatchSpanProcessor(local_exporter))
                except Exception as le:
                    logger.debug(f"Local Phoenix exporter init skipped: {le}")

            trace.set_tracer_provider(self.provider)
            self.tracer = trace.get_tracer("aura-phoenix-mas", "1.0.0")
            logger.info(f"Arize Phoenix Tracer initialized: project='{self.project_name}', endpoint='{self.endpoint}', local_port={self.local_port}")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenTelemetry provider: {e}")
            self.tracer = None

    def start_agent_span(
        self,
        step_id: str,
        thread_id: str,
        agent_name: str,
        input_data: Any
    ) -> Optional[Any]:
        """
        Starts an OpenTelemetry span for an agent or tool execution step.
        Supports automatic parent nesting under thread root pipeline.
        """
        if not self.tracer:
            return None

        try:
            # Determine OpenInference span kind and attributes
            is_root = agent_name in ("Conversational Pipeline", "Sequential MAS Pipeline")
            is_tool = "search" in agent_name.lower() or "tool" in agent_name.lower()

            span_kind = "chain" if is_root else ("tool" if is_tool else "agent")

            # Check if there is an active parent context for this thread
            parent_ctx = None
            if not is_root and thread_id in self.root_contexts:
                parent_ctx = self.root_contexts[thread_id]

            span = self.tracer.start_span(
                name=agent_name,
                context=parent_ctx
            )

            # Set OpenInference and MAS attributes
            span.set_attribute("openinference.span.kind", span_kind)
            span.set_attribute("openinference.project.name", self.project_name)
            span.set_attribute("session.id", str(thread_id))
            span.set_attribute("step_id", str(step_id))
            span.set_attribute("agent.name", str(agent_name))

            if is_tool:
                span.set_attribute("tool.name", "mongodb_atlas_hybrid_search")

            input_summary = str(input_data)[:2000] if input_data else ""
            span.set_attribute("input.value", input_summary)
            span.set_attribute("input.mime_type", "text/plain")

            self.active_spans[step_id] = span

            # If this is root, save context for child spans
            if is_root:
                self.root_contexts[thread_id] = trace.set_span_in_context(span)

            return span
        except Exception as e:
            logger.debug(f"Failed to start span for {agent_name}: {e}")
            return None

    def end_agent_span(
        self,
        step_id: str,
        output_data: Any,
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ) -> None:
        """
        Completes an active OpenTelemetry span, capturing token consumption and output payload.
        """
        if not self.tracer:
            return

        span = self.active_spans.pop(step_id, None)
        if not span:
            return

        try:
            output_summary = str(output_data)[:2000] if output_data else ""
            span.set_attribute("output.value", output_summary)
            span.set_attribute("output.mime_type", "text/plain")

            # Token usage conventions
            if prompt_tokens or completion_tokens:
                span.set_attribute("llm.token_count.prompt", prompt_tokens)
                span.set_attribute("llm.token_count.completion", completion_tokens)
                span.set_attribute("llm.token_count.total", prompt_tokens + completion_tokens)

            span.set_status(Status(StatusCode.OK))
            span.end()

            self.trace_counter += 1
            self.last_trace_time = time.time()
        except Exception as e:
            logger.debug(f"Failed to end span {step_id}: {e}")

    def record_agent_error(self, step_id: str, error: Exception) -> None:
        """Records error status and exception stack on active span."""
        if not self.tracer:
            return

        span = self.active_spans.pop(step_id, None)
        if not span:
            return

        try:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
            span.end()
        except Exception as e:
            logger.debug(f"Failed to record span error {step_id}: {e}")

    def flush(self, timeout_millis: int = 2000) -> None:
        """Forces flush on trace provider processors."""
        if self.provider and hasattr(self.provider, "force_flush"):
            try:
                self.provider.force_flush(timeout_millis=timeout_millis)
            except Exception:
                pass

    def get_status(self) -> Dict[str, Any]:
        """Diagnostic summary of Phoenix tracing configuration and counters."""
        return {
            "enabled": self.enabled,
            "opentelemetry_installed": OPENTELEMETRY_AVAILABLE,
            "active": self.tracer is not None,
            "project_name": self.project_name,
            "cloud_endpoint": self.endpoint,
            "local_port": self.local_port,
            "total_traces_emitted": self.trace_counter,
            "last_trace_timestamp": self.last_trace_time,
            "active_in_flight_spans": len(self.active_spans)
        }


# Global singleton instance
_phoenix_tracer: Optional[PhoenixTracer] = None

def get_phoenix_tracer() -> PhoenixTracer:
    """Returns singleton PhoenixTracer instance."""
    global _phoenix_tracer
    if _phoenix_tracer is None:
        _phoenix_tracer = PhoenixTracer()
    return _phoenix_tracer

