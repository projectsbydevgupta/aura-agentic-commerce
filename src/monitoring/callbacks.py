"""
AURA Task Monitoring & Execution Callback System
Implements Criterion 5:
- Tracks intermediate agent outputs, execution flows, and errors
- Captures start/end timestamps, latency in milliseconds
- Tracks prompt/completion token consumption per agent
- Collects metrics matrix to power MAS Heat Map visualizations
"""

import time
import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Arize Phoenix OpenTelemetry Bridge
def _get_tracer():
    try:
        from src.monitoring.phoenix_tracer import get_phoenix_tracer
        return get_phoenix_tracer()
    except Exception:
        return None

# =====================================================================
# Metric Data Records
# =====================================================================

class AgentStepLog(BaseModel):
    step_id: str
    thread_id: str
    agent_name: str
    start_time: float
    end_time: Optional[float] = None
    latency_ms: float = 0.0
    input_summary: str = ""
    output_summary: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    status: str = "running"  # running, completed, error, retried
    error_message: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = []


class MASAgentSummary(BaseModel):
    agent_name: str
    total_invocations: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    avg_tokens_per_call: float = 0.0
    error_count: int = 0
    retry_count: int = 0


# =====================================================================
# Agent Monitoring Callback Handler
# =====================================================================

class AgentExecutionMonitorCallback:
    """
    Central monitoring and auditing callback engine.
    Logs execution flows across Query, Context, Search, Validation, Upsell, and Campaign agents.
    """

    def __init__(self, log_dir: str = "data/monitoring_logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.active_steps: Dict[str, AgentStepLog] = {}
        self.completed_logs: List[AgentStepLog] = []

    def on_agent_start(
        self,
        step_id: str,
        thread_id: str,
        agent_name: str,
        input_data: Any
    ) -> AgentStepLog:
        """Called when an agent node begins execution."""
        summary = str(input_data)[:200] if input_data else ""
        step = AgentStepLog(
            step_id=step_id,
            thread_id=thread_id,
            agent_name=agent_name,
            start_time=time.time(),
            input_summary=summary,
            status="running"
        )
        self.active_steps[step_id] = step

        # Dispatch OpenTelemetry span to Arize Phoenix
        tracer = _get_tracer()
        if tracer:
            try:
                tracer.start_agent_span(step_id, thread_id, agent_name, input_data)
            except Exception:
                pass

        return step

    def on_agent_end(
        self,
        step_id: str,
        output_data: Any,
        prompt_tokens: int = 0,
        completion_tokens: int = 0
    ) -> Optional[AgentStepLog]:
        """Called when an agent completes execution successfully."""
        step = self.active_steps.pop(step_id, None)
        if not step:
            return None

        step.end_time = time.time()
        step.latency_ms = round((step.end_time - step.start_time) * 1000, 2)
        step.output_summary = str(output_data)[:200] if output_data else ""
        step.prompt_tokens = prompt_tokens
        step.completion_tokens = completion_tokens
        step.total_tokens = prompt_tokens + completion_tokens
        step.status = "completed"

        self.completed_logs.append(step)
        self._persist_log_entry(step)

        # Dispatch span completion to Arize Phoenix
        tracer = _get_tracer()
        if tracer:
            try:
                tracer.end_agent_span(step_id, output_data, prompt_tokens, completion_tokens)
            except Exception:
                pass

        return step

    def on_agent_error(
        self,
        step_id: str,
        error: Exception,
        retry_flag: bool = False
    ) -> Optional[AgentStepLog]:
        """Called when an agent encounters an error or requires retry."""
        step = self.active_steps.pop(step_id, None)
        if not step:
            return None

        step.end_time = time.time()
        step.latency_ms = round((step.end_time - step.start_time) * 1000, 2)
        step.status = "retried" if retry_flag else "error"
        step.error_message = str(error)

        self.completed_logs.append(step)
        self._persist_log_entry(step)

        # Dispatch span error to Arize Phoenix
        tracer = _get_tracer()
        if tracer:
            try:
                tracer.record_agent_error(step_id, error)
            except Exception:
                pass

        return step

    def on_tool_call(
        self,
        step_id: str,
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
        duration_ms: float
    ):
        """Attaches tool execution audit to the current active agent step."""
        step = self.active_steps.get(step_id)
        if step:
            step.tool_calls.append({
                "tool_name": tool_name,
                "input": str(tool_input)[:150],
                "output_preview": str(tool_output)[:150],
                "duration_ms": duration_ms,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

    def get_agent_metrics_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Calculates aggregated metrics matrix for all MAS agents.
        Returns data formatted directly for generating the MAS Heat Map:
        {
          "QueryParser": {"p50_latency_ms": 250, "prompt_tokens": 320, ...},
          "CatalogSearch": {...},
          ...
        }
        """
        agents = [
            "Query Agent", "Context Agent", "Search Node",
            "Validation Agent", "Upsell Agent", "Campaign Agent"
        ]
        
        matrix = {}
        for agent in agents:
            logs = [log for log in self.completed_logs if log.agent_name.lower() in agent.lower()]
            if not logs:
                # Provide baseline representative values if fresh run
                matrix[agent] = {
                    "total_calls": 0,
                    "avg_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "avg_prompt_tokens": 0.0,
                    "avg_completion_tokens": 0.0,
                    "error_rate": 0.0
                }
                continue

            latencies = sorted([l.latency_ms for l in logs])
            p50 = latencies[len(latencies) // 2]
            p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) > 1 else p50
            avg_prompt = sum(l.prompt_tokens for l in logs) / len(logs)
            avg_comp = sum(l.completion_tokens for l in logs) / len(logs)
            err_count = sum(1 for l in logs if l.status in ["error", "retried"])

            matrix[agent] = {
                "total_calls": len(logs),
                "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
                "p95_latency_ms": round(p95, 1),
                "avg_prompt_tokens": round(avg_prompt, 1),
                "avg_completion_tokens": round(avg_comp, 1),
                "error_rate": round(err_count / len(logs), 2)
            }

        return matrix

    def _persist_log_entry(self, step: AgentStepLog):
        """Appends log entry to daily audit file (supports Pydantic v1 & v2)."""
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filepath = os.path.join(self.log_dir, f"mas_execution_{today}.jsonl")
            payload = step.model_dump_json() if hasattr(step, "model_dump_json") else step.json()
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(payload + "\n")
        except Exception as e:
            print(f"Monitoring log write notice: {e}")


# Singleton instance
_monitor_callback = AgentExecutionMonitorCallback()

def get_monitor_callback() -> AgentExecutionMonitorCallback:
    """Returns global AgentExecutionMonitorCallback singleton."""
    return _monitor_callback
