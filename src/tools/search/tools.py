import os
from typing import Dict, Any

from dotenv import load_dotenv

from tools.base import BaseAgentTool

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Only instantiate if key is available
_tavily = None
if TAVILY_API_KEY:
    from langchain_tavily import TavilySearch
    _tavily = TavilySearch(tavily_api_key=TAVILY_API_KEY, max_results=3)


class WebSearchTool(BaseAgentTool):
    """Search the web for current information on architecture topics."""

    name: str = "search_web"
    description: str = "Search the web for information related to the query."
    categories: list = ["search", "general_question"]

    def _run(self, query: str) -> Dict[str, Any]:  # type: ignore[override]
        if _tavily is None:
            return {"error": "TAVILY_API_KEY not set — web search unavailable.", "results": []}
        try:
            return _tavily.invoke(query)
        except Exception as e:
            return {"error": str(e), "results": []}


# Module-level instance — import this directly for use in nodes
search_web = WebSearchTool()
