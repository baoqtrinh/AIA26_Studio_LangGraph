"""
GH Tool execution node — use_tool branch.

The LLM picks which loaded MCP tool to call and with what args, then
formats the result as a natural-language answer.

Vision tools (capture_viewport, etc.)
--------------------------------------
If a tool result is a JSON object that contains an "image_base64" key,
the node automatically forwards it to the vision-capable LLM so a VLM
can reason about the rendered scene.
"""
import json
import textwrap
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from config.prompts import build_csharp_system_prompt
from models.state import BoxState
from utils.llm_utils import chat_llm, reason_about_image

_HR = "─" * 72


def _think(label: str, text: str):
    prefix = f"  ┊ {label}: "
    body = str(text).strip().replace("\n", " ")
    for i, line in enumerate(textwrap.wrap(body, width=68)):
        print((prefix if i == 0 else " " * len(prefix)) + line)


def _reason_about_image(base64_png: str, user_input: str, view_name: str = "") -> str:
    """Delegate to the shared utility in llm_utils."""
    print(f"  ┊ image captured — asking VLM to reason about the scene...")
    question = (
        f"This is a screenshot of the current Rhino 3D viewport.\n"
        f"Original user request: \"{user_input}\"\n\n"
        "Describe what you see: geometry types, approximate sizes, any issues "
        "or suggestions for the design. Be concise."
    )
    return reason_about_image(base64_png, question=question, view_name=view_name)


def _handle_image_result(result_str: str, user_input: str) -> str:
    """If result_str is a JSON payload with image_base64, run VLM reasoning and return analysis."""
    try:
        data = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return result_str  # not JSON, return as-is

    if not isinstance(data, dict) or "image_base64" not in data:
        return result_str

    base64_png = data["image_base64"]
    view_name  = data.get("view_name", "")
    w, h       = data.get("width", "?"), data.get("height", "?")
    analysis   = _reason_about_image(base64_png, user_input, view_name)

    return f"[Viewport capture {w}×{h} — {view_name}]\n\n{analysis}"


def execute_gh_tool_fn(state: BoxState) -> BoxState:
    """Use LLM + tool-calling to invoke the appropriate GH MCP tool."""
    # Lazily import to avoid circular deps and to pick up any reload
    from tools import TOOL_CLASSES

    user_input: str = state.request.get("user_input", "")

    if not TOOL_CLASSES:
        state.answer = (
            "No GH tools are currently loaded. "
            "Make sure the Grasshopper plugin is running and set to Active, "
            "then reload tools in the sidebar."
        )
        state.done = True
        state.history.append({"node": "execute_gh_tool", "status": "no_tools"})
        return state

    # Build tool descriptions for the system prompt
    tool_list = "\n".join(
        f"- `{t.name}`: {t.description}" for t in TOOL_CLASSES
    )
    tool_names = [t.name for t in TOOL_CLASSES]
    print(f"  ┊ tools available: {', '.join(tool_names)}")

    has_csharp = "run_csharp_script" in tool_names
    base = (
        "You are a Rhino/Grasshopper design assistant. "
        "Pick the most relevant tool, supply the required arguments, and call it. "
        "If several tools are needed, call them in sequence."
    )
    system_prompt = (
        build_csharp_system_prompt(base, tool_list)
        if has_csharp
        else f"{base}\n\nAvailable tools:\n{tool_list}"
    )

    llm_with_tools = chat_llm.bind_tools(TOOL_CLASSES)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input),
    ]

    # First LLM call — use .invoke() so RunnableBinding (Gemini) passes tools correctly
    print(f"  ┊ asking LLM which tool to call...")
    ai_msg = llm_with_tools.invoke(messages)

    if ai_msg.content:
        _think("LLM thought", ai_msg.content)

    results: Dict[str, Any] = {}

    if not ai_msg.tool_calls:
        # LLM answered without calling a tool
        state.answer = ai_msg.content
        state.done = True
        state.history.append({"node": "execute_gh_tool", "status": "direct_answer"})
        return state

    # Execute each requested tool call
    tool_messages: List[ToolMessage] = []
    for tc in ai_msg.tool_calls:
        tool_name: str = tc["name"]
        tool_args: Dict[str, Any] = tc.get("args", {})
        tool_id: str = tc.get("id", tool_name)

        args_str = ", ".join(f"{k}={v}" for k, v in tool_args.items())
        print(f"  ┊ calling tool: {tool_name}({args_str})")

        matching = [t for t in TOOL_CLASSES if t.name == tool_name]
        if not matching:
            result_str = f"Error: tool '{tool_name}' not found."
        else:
            result_str = matching[0]._run(**tool_args)

        # ── Vision result: send image to VLM instead of raw base64 ───────────
        result_str = _handle_image_result(result_str, user_input)

        _think(f"{tool_name} result", result_str)

        results[tool_name] = result_str
        tool_messages.append(
            ToolMessage(content=result_str, tool_call_id=tool_id)
        )

    # Second LLM call — synthesise results into natural language
    print(f"  ┊ synthesising answer...")
    followup_messages = [*messages, ai_msg, *tool_messages]
    final_answer = chat_llm.invoke(followup_messages).content

    state.answer = final_answer
    state.tool_results = results
    state.done = True
    state.history.append({
        "node": "execute_gh_tool",
        "tools_called": list(results.keys()),
        "results": results,
    })
    return state
