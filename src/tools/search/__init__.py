"""
tools/search — Web search tool (Tavily-backed).

Usage
-----
    from tools.search import search_web
    result = search_web.invoke("passive solar design principles")
"""

from .tools import WebSearchTool, search_web

__all__ = ["WebSearchTool", "search_web"]
