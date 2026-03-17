"""
Tools package.

Sub-packages
────────────
  tools/mcp/       Dynamic tools loaded from the GH MCP Server (Grasshopper).
  tools/search/    Web-search tool (Tavily-backed).
  tools/building/  Static building-calculation helpers (design_building branch).
  tools/base.py    BaseAgentTool — subclass this when adding a new tool.

Public API (backward-compatible)
─────────────────────────────────
  TOOL_CLASSES        List[BaseTool]  auto-loaded from MCP server at startup
  load_mcp_tools()    → List[BaseTool]
  reload_mcp_tools()  → List[BaseTool]  re-fetches from running GH server
"""

from .mcp import TOOL_CLASSES, DynamicMCPTool, load_mcp_tools, reload_mcp_tools

__all__ = [
    "TOOL_CLASSES",
    "DynamicMCPTool",
    "load_mcp_tools",
    "reload_mcp_tools",
]
