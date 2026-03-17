"""
list_tools.py
─────────────
Standalone script — run directly to print every tool exposed by the
running GH MCP Server (the Grasshopper plugin).

Usage (from any directory):
    python tools/mcp/list_tools.py
    python tools/mcp/list_tools.py --endpoint http://localhost:5100
"""

import argparse
import sys
import os

# ── Ensure AgentApp root is importable when run as a script ────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))   # .../AgentApp/tools/mcp
_ROOT = os.path.dirname(os.path.dirname(_HERE))       # .../AgentApp
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from app.config import MCP_GH_ENDPOINT, MCP_TIMEOUT
except ImportError:
    MCP_GH_ENDPOINT = os.getenv("MCP_GH_ENDPOINT", "http://localhost:5100")
    MCP_TIMEOUT = int(os.getenv("MCP_TIMEOUT", "30"))

import requests


def list_tools(endpoint: str, timeout: int) -> None:
    # ── Health check ──────────────────────────────────────────────────────────
    try:
        health = requests.get(f"{endpoint}/api/health", timeout=timeout)
        health.raise_for_status()
        h = health.json()
        print(f"Server  : {endpoint}")
        print(f"Status  : {h.get('status', '?')}   Tools registered: {h.get('tools', '?')}")
        print()
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to GH MCP Server at {endpoint}")
        print("       Make sure Rhino/Grasshopper is running with the MCPServer component active.")
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR during health check: {exc}")
        sys.exit(1)

    # ── Fetch tool list ───────────────────────────────────────────────────────
    try:
        resp = requests.get(f"{endpoint}/api/list_tools", timeout=timeout)
        resp.raise_for_status()
        tools = resp.json().get("tools", [])
    except Exception as exc:
        print(f"ERROR fetching tool list: {exc}")
        sys.exit(1)

    if not tools:
        print("No tools registered on the server.")
        return

    sep = "─" * 60

    for i, tool in enumerate(tools, 1):
        name        = tool.get("name", "(unnamed)")
        description = tool.get("description", "(no description)")
        schema      = tool.get("inputSchema", {})
        props       = schema.get("properties", {})
        required    = set(schema.get("required", []))

        print(sep)
        print(f"[{i}] {name}")
        print()
        print("  Description:")
        for line in description.splitlines():
            print(f"    {line}")
        print()

        if props:
            print("  Inputs:")
            for param, meta in props.items():
                req_marker = " *" if param in required else "  "
                desc = meta.get("description", "")
                typ  = meta.get("type", "string")
                print(f"   {req_marker} {param}  ({typ})  — {desc}")
        else:
            print("  Inputs: (none)")

        outputs = tool.get("outputs", {})
        if outputs:
            print("  Outputs:")
            for out_name, out_type in outputs.items():
                print(f"      {out_name}  ({out_type})")
        else:
            print("  Outputs: (none)")

        categories = tool.get("categories", [])
        if categories:
            print(f"  Categories: {', '.join(categories)}")

        print()

    print(sep)
    print(f"Total: {len(tools)} tool(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List tools from the GH MCP Server")
    parser.add_argument(
        "--endpoint",
        default=MCP_GH_ENDPOINT,
        help=f"MCP server base URL (default: {MCP_GH_ENDPOINT})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=MCP_TIMEOUT,
        help=f"Request timeout in seconds (default: {MCP_TIMEOUT})",
    )
    args = parser.parse_args()
    list_tools(args.endpoint, args.timeout)
