"""
Base class for all Python-side agent tools.

Subclass `BaseAgentTool` (instead of `BaseTool` directly) to guarantee every
tool in this project:

  - declares a ``categories`` list that indicates which graph branches use it
  - provides a default async implementation (delegates to ``_run``)

Example
-------
    from tools.base import BaseAgentTool

    class MyTool(BaseAgentTool):
        name: str = "my_tool"
        description: str = "Does something useful."
        categories: list[str] = ["tool_use"]

        def _run(self, *, param: str) -> str:
            return f"result for {param}"

Notes
-----
- ``BaseTool`` (from LangChain) already marks ``_run`` abstract, so every
  concrete subclass must implement it.
- ``args_schema`` (a Pydantic model) is recommended for structured inputs.
  See ``tools/mcp/loader.py`` for an example of building one from JSON Schema.
"""

from typing import List

from langchain_core.tools import BaseTool


class BaseAgentTool(BaseTool):
    """Abstract base for all tools used by this agent.

    Add any shared validation, logging, or error-handling here and every
    concrete tool will inherit it automatically.
    """

    categories: List[str] = []

    async def _arun(self, **kwargs) -> str:  # type: ignore[override]
        """Default async implementation — runs the sync version."""
        return self._run(**kwargs)
