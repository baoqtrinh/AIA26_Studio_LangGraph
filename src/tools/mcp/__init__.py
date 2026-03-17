"""
tools/mcp — Dynamic MCP tool loader for the GH MCP Server.

Public API
----------
    TOOL_CLASSES        List[BaseTool]  — auto-loaded at import (empty if server offline)
    load_mcp_tools()    → List[BaseTool]
    reload_mcp_tools()  → List[BaseTool]  (re-fetches at runtime)
    DynamicMCPTool      — the concrete tool class (subclass of BaseAgentTool)
"""

from .loader import TOOL_CLASSES, DynamicMCPTool, load_mcp_tools, reload_mcp_tools

__all__ = [
    "TOOL_CLASSES",
    "DynamicMCPTool",
    "load_mcp_tools",
    "reload_mcp_tools",
]
