"""
Dynamically load tools from the GH MCP Server (Grasshopper plugin HTTP server).
Adapted from GlabAgents pattern — single source of truth is the running server.
"""
import json
import logging
from typing import Any, Dict, List, Optional

import requests
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from tools.base import BaseAgentTool

# Import config from app/ package; fall back to env vars if run stand-alone
from settings import MCP_GH_ENDPOINT, MCP_TIMEOUT

logger = logging.getLogger(__name__)


# ── Dynamic tool wrapper ──────────────────────────────────────────────────────

class DynamicMCPTool(BaseAgentTool):
    """LangChain tool that forwards calls to the GH MCP HTTP server.

    Instances are created automatically at import time (and on reload) from
    the tool definitions served by the running Grasshopper plugin.
    Do NOT instantiate this directly — use ``load_mcp_tools()`` instead.
    """

    mcp_tool_name: str
    mcp_endpoint: str
    mcp_timeout: int
    categories: List[str] = []

    def _run(self, **kwargs: Any) -> str:
        clean_args = {k: v for k, v in kwargs.items() if v is not None}
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": self.mcp_tool_name, "arguments": clean_args},
        }
        try:
            resp = requests.post(
                self.mcp_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.mcp_timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                return f"Error: {data['error']}"
            content = data.get("result", {}).get("content", [])
            if not content:
                return "No result returned"
            texts = [item["text"] for item in content if item.get("type") == "text"]
            result_str = "\n".join(texts) if texts else str(content)
            return result_str
        except requests.exceptions.Timeout:
            return f"Error: request timed out after {self.mcp_timeout}s"
        except requests.exceptions.ConnectionError:
            return "Error: cannot connect to GH MCP Server. Is the Grasshopper plugin running?"
        except Exception as exc:
            return f"Error calling tool: {exc}"

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


# ── Schema conversion ─────────────────────────────────────────────────────────

def convert_json_schema_to_pydantic(tool_name: str, schema: Dict[str, Any]) -> type:
    """Build a Pydantic model from a JSON Schema dict."""
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    type_map = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    fields: Dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        py_type = type_map.get(prop_schema.get("type", "string"), str)
        desc = prop_schema.get("description", "")
        if prop_name in required:
            fields[prop_name] = (py_type, Field(description=desc))
        else:
            fields[prop_name] = (Optional[py_type], Field(default=None, description=desc))

    if not fields:
        fields["no_params"] = (Optional[str], Field(default=None, description="No parameters needed"))

    model_name = "".join(w.title() for w in tool_name.split("_")) + "Input"
    return create_model(model_name, **fields)


def create_tool_from_definition(tool_def: Dict[str, Any]) -> BaseTool:
    name = tool_def.get("name", "unknown_tool")
    description = tool_def.get("description", "No description")
    input_schema = tool_def.get("inputSchema", {})
    categories = tool_def.get("categories", ["grasshopper", "custom"])
    pydantic_model = convert_json_schema_to_pydantic(name, input_schema)
    return DynamicMCPTool(
        name=name,
        description=description,
        args_schema=pydantic_model,
        mcp_tool_name=name,
        mcp_endpoint=MCP_GH_ENDPOINT,
        mcp_timeout=MCP_TIMEOUT,
        categories=categories,
    )


# ── Fetching ──────────────────────────────────────────────────────────────────

def fetch_tool_definitions() -> List[Dict[str, Any]]:
    """Query tools/list from the running GH MCP server (JSON-RPC 2.0)."""
    try:
        resp = requests.post(
            MCP_GH_ENDPOINT,
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            headers={"Content-Type": "application/json"},
            timeout=MCP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        tools = data.get("result", {}).get("tools", [])
        logger.info(f"GH MCP Server: loaded {len(tools)} tools")
        return tools
    except requests.exceptions.ConnectionError:
        logger.warning(
            f"GH MCP Server not reachable at {MCP_GH_ENDPOINT}. "
            "Start the Grasshopper plugin to enable GH tools."
        )
        return []
    except Exception as exc:
        logger.warning(f"Failed to fetch tools from GH MCP Server: {exc}")
        return []


def load_mcp_tools() -> List[BaseTool]:
    """Load all GH tools from the MCP server as LangChain tools."""
    tool_defs = fetch_tool_definitions()
    tools: List[BaseTool] = []
    for td in tool_defs:
        try:
            tool = create_tool_from_definition(td)
            tools.append(tool)
            logger.info(f"  + {tool.name}")
        except Exception as exc:
            logger.error(f"  - Failed to load '{td.get('name', '?')}': {exc}")
    return tools


def reload_mcp_tools() -> List[BaseTool]:
    """Re-fetch tools at runtime (call when user hits 'Reload Tools' in sidebar)."""
    global TOOL_CLASSES
    TOOL_CLASSES = load_mcp_tools()
    return TOOL_CLASSES


# Auto-load at import — gracefully empty if server is offline
TOOL_CLASSES: List[BaseTool] = load_mcp_tools()
