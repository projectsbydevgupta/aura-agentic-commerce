"""
AURA Model Context Protocol (MCP) Client
Implements Criterion 2:
- Allows agents to dynamically discover tool declarations
- Invokes MCP tools using standard message format
- Unpacks structured responses for LLM reasoning
"""

import json
from typing import Dict, Any, List, Optional
from src.mcp.mcp_server import get_mcp_server

class MCPClient:
    """
    Standard MCP Client for Agent-to-Tool communication.
    Discovers tool specifications and executes invocations.
    """

    def __init__(self, server=None):
        self.server = server or get_mcp_server()

    def discover_tools(self) -> List[Dict[str, Any]]:
        """Queries the MCP server for available tool schemas."""
        return self.server.list_tools()

    def invoke_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a tool on the MCP server and parses the JSON response.
        """
        raw_response = self.server.call_tool(tool_name, arguments)
        
        if raw_response.get("isError"):
            err_msg = raw_response["content"][0]["text"]
            raise RuntimeError(f"MCP Tool Execution Failed: {err_msg}")

        # Parse the structured text output
        text_content = raw_response["content"][0]["text"]
        try:
            return json.loads(text_content)
        except json.JSONDecodeError:
            return {"raw_text": text_content}

    # Helper convenience methods for agents
    def search_catalog(
        self,
        query: str,
        gender: Optional[str] = None,
        category: Optional[str] = None,
        max_budget_inr: Optional[float] = None,
        limit: int = 8
    ) -> List[Dict[str, Any]]:
        """Convenience method for searching the catalog via MCP."""
        args = {"query": query, "limit": limit}
        if gender: args["gender"] = gender
        if category: args["category"] = category
        if max_budget_inr: args["max_budget_inr"] = max_budget_inr
        
        res = self.invoke_tool("catalog_hybrid_search", args)
        return res.get("products", [])

    def calculate_fx(
        self,
        amount: float,
        source_currency: str = "INR",
        target_currency: str = "INR",
        coupon_code: Optional[str] = None
    ) -> Dict[str, Any]:
        """Convenience method for FX and voucher math via MCP."""
        args = {
            "amount": amount,
            "source_currency": source_currency,
            "target_currency": target_currency
        }
        if coupon_code:
            args["coupon_code"] = coupon_code
            
        return self.invoke_tool("financial_fx_calculator", args)

