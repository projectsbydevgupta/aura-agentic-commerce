"""
Unit & Integration Tests for Model Context Protocol (MCP) Server & Client.
Validates Criterion 2 (Tool Integration using MCP: at least 2 external tools).
"""

import unittest
from src.mcp.mcp_server import get_mcp_server
from src.mcp.mcp_client import MCPClient

class TestMCPIntegration(unittest.TestCase):

    def setUp(self):
        self.server = get_mcp_server()
        self.client = MCPClient(server=self.server)

    def test_01_mcp_list_tools(self):
        """Tests that MCP server lists at least two declared tools with inputSchema."""
        tools = self.client.discover_tools()
        self.assertGreaterEqual(len(tools), 2)
        tool_names = [t["name"] for t in tools]
        self.assertIn("catalog_hybrid_search", tool_names)
        self.assertIn("financial_fx_calculator", tool_names)

        # Verify schema structure
        for tool in tools:
            self.assertIn("name", tool)
            self.assertIn("description", tool)
            self.assertIn("inputSchema", tool)

    def test_02_mcp_financial_fx_calculator(self):
        """Tests MCP Tool 2: Currency normalization, tax, and voucher calculation."""
        res = self.client.calculate_fx(
            amount=5000.0,
            source_currency="INR",
            target_currency="INR",
            coupon_code="AURA20"
        )
        self.assertEqual(res.get("status"), "success")
        self.assertEqual(res.get("original_amount"), 5000.0)
        self.assertEqual(res.get("discount_percentage"), 20)
        self.assertEqual(res.get("discount_savings_inr"), 1000.0)
        self.assertEqual(res.get("net_payable"), 4000.0)

    def test_03_mcp_financial_fx_usd_to_inr(self):
        """Tests MCP Tool 2: Converting USD to INR ($100 * 50 = ₹5000)."""
        res = self.client.calculate_fx(
            amount=100.0,
            source_currency="USD",
            target_currency="INR"
        )
        self.assertEqual(res.get("status"), "success")
        self.assertEqual(res.get("net_payable"), 5000.0)

    def test_04_mcp_invalid_tool_error_handling(self):
        """Tests that unknown tool returns an MCP error JSON structure."""
        res = self.server.call_tool("non_existent_tool", {})
        self.assertTrue(res.get("isError"))
        self.assertIn("not found", res["content"][0]["text"])

    def test_05_mcp_catalog_hybrid_search_output_validation(self):
        """Tests that catalog_hybrid_search returns items with price and passes tool output validation."""
        tool_res = self.server.execute_tool("catalog_hybrid_search", {"query": "black shirt", "limit": 3})
        self.assertIn("products", tool_res)
        self.assertTrue(tool_res["validation_passed"])
        self.assertIn("mock", tool_res)
        if tool_res["products"]:
            first = tool_res["products"][0]
            self.assertIn("price", first)
            self.assertIn("price_inr", first)
            self.assertIn("price_usd", first)

if __name__ == "__main__":
    unittest.main()

