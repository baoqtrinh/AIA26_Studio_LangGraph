"""Plan execution nodes — multi-tool chaining (plan branch).

Flow
────
planner_fn       Ask the LLM to break the user's request into an ordered list
                 of tool calls (JSON). No concrete args yet.

plan_step_fn     Execute the current step: give the LLM the step intent +
                 all previous results → it makes a tool call → store result →
                 advance the counter.  Looped by plan_step_router.

plan_step_router Return "continue" when more steps remain, "done" otherwise.

plan_summary_fn  Synthesise all step results into a final natural-language
                 answer for the user.
"""
import json
import re
import textwrap
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts import build_csharp_system_prompt
from models.state import BoxState
from utils.llm_utils import chat_llm, fast_llm

_HR = "─" * 72


def _think(label: str, text: str):
    prefix = f"  ┊ {label}: "
    body = str(text).strip().replace("\n", " ")
    for i, line in enumerate(textwrap.wrap(body, width=68)):
        print((prefix if i == 0 else " " * len(prefix)) + line)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Planner — produce the step list
# ─────────────────────────────────────────────────────────────────────────────

def planner_fn(state: BoxState) -> BoxState:
    """Decompose the user's multi-step request into an ordered tool-call plan."""
    from tools import TOOL_CLASSES

    user_input: str = state.request.get("user_input", "")

    if not TOOL_CLASSES:
        state.answer = (
            "No GH tools are loaded. "
            "Start the Grasshopper plugin and run 'reload'."
        )
        state.done = True
        return state

    tool_descriptions = "\n".join(
        f"- {t.name}: {t.description}" for t in TOOL_CLASSES
    )

    prompt = f"""You are a planning assistant for a Grasshopper 3D-modelling agent.
The user wants to perform a multi-step design task.
Break it into an ordered list of Grasshopper tool calls.

Available tools:
{tool_descriptions}

User request: "{user_input}"

Respond with ONLY a valid JSON array — no markdown, no extra text. Each item:
  "step"       : integer starting at 1
  "tool"       : exact tool name from the list above
  "intent"     : one sentence describing what this step accomplishes
  "output_key" : short camelCase key to reference this result in later steps

Example:
[
  {{"step": 1, "tool": "draw_box", "intent": "Draw the rectangular building base", "output_key": "base"}},
  {{"step": 2, "tool": "draw_cylinder", "intent": "Add a cylindrical tower on top of the base", "output_key": "tower"}}
]

Plan:"""

    print(f"\n{_HR}")
    print(f"  ▶  PLAN MODE  —  decomposing task...")
    print(_HR)
    _think("request", user_input)

    try:
        raw = fast_llm(prompt)
    except Exception as exc:
        state.answer = f"Planner LLM error: {exc}"
        state.done = True
        return state

    raw_str = str(raw).strip()
    # Strip markdown code fences if present
    raw_str = re.sub(r"^```(?:json)?\s*", "", raw_str)
    raw_str = re.sub(r"\s*```$", "", raw_str)

    plan = None
    try:
        plan = json.loads(raw_str)
        assert isinstance(plan, list) and len(plan) > 0
    except Exception:
        m = re.search(r"\[.*?\]", raw_str, re.DOTALL)
        if m:
            try:
                plan = json.loads(m.group())
            except Exception:
                pass

    if not plan:
        state.answer = f"Could not parse a plan from the LLM output:\n{raw_str}"
        state.done = True
        return state

    state.plan = plan
    state.plan_step = 0
    state.plan_results = {}

    print(f"  ┊ {len(plan)}-step plan:")
    for s in plan:
        print(f"  ┊   [{s.get('step','?')}] {s.get('tool','?')}  —  {s.get('intent','')}")

    state.history.append({"node": "planner", "plan": plan})
    return state


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Step executor — looped once per plan step
# ─────────────────────────────────────────────────────────────────────────────

def plan_step_fn(state: BoxState) -> BoxState:
    """Execute the current plan step via LLM tool-calling, then advance the counter."""
    from tools import TOOL_CLASSES

    plan = state.plan or []
    idx = state.plan_step

    # Safety guard (should not happen)
    if idx >= len(plan):
        return state

    step = plan[idx]
    total = len(plan)
    step_num   = step.get("step", idx + 1)
    intent     = step.get("intent", "")
    target_tool = step.get("tool", "")
    output_key  = step.get("output_key", f"step_{step_num}")

    print(f"\n  ── Step {step_num}/{total}: {intent}")

    # ── Context from previous steps ──────────────────────────────────────────
    prev_context = ""
    if state.plan_results:
        lines = [f"  • {k}: {v}" for k, v in state.plan_results.items()]
        prev_context = "\nResults from previous steps:\n" + "\n".join(lines)

    tool_list = "\n".join(f"- {t.name}: {t.description}" for t in TOOL_CLASSES)

    base = (
        f"You are executing step {step_num} of {total} in a multi-step Rhino/Grasshopper design plan.\n"
        f"Step intent: {intent}\n"
        f"Preferred tool: {target_tool}\n"
        f"{prev_context}\n\n"
        "Call the correct tool with concrete numeric arguments based on the user's original "
        "request and any relevant previous step results. "
        "Do NOT use placeholder strings — only real values."
    )
    tool_names = [t.name for t in TOOL_CLASSES]
    prompt_content = (
        build_csharp_system_prompt(base, tool_list)
        if target_tool == "run_csharp_script" or "run_csharp_script" in tool_names
        else f"{base}\n\nAvailable tools:\n{tool_list}"
    )
    system_msg = SystemMessage(content=prompt_content)
    user_msg = HumanMessage(content=state.request.get("user_input", ""))

    llm_with_tools = chat_llm.bind_tools(TOOL_CLASSES)
    print(f"  ┊ asking LLM for tool arguments...")
    response = llm_with_tools._generate([system_msg, user_msg])
    ai_msg = response.generations[0].message

    if ai_msg.content:
        _think("LLM thought", ai_msg.content)

    # ── Execute tool call ─────────────────────────────────────────────────────
    if not ai_msg.tool_calls:
        result_str = ai_msg.content or "(no output — LLM did not call a tool)"
        _think(f"step {step_num}", result_str)
    else:
        tc = ai_msg.tool_calls[0]          # honour first call; LLM should only pick one per step
        tool_name = tc["name"]
        tool_args: Dict[str, Any] = tc.get("args", {})
        args_str = ", ".join(f"{k}={v}" for k, v in tool_args.items())
        print(f"  ┊ calling: {tool_name}({args_str})")

        matching = [t for t in TOOL_CLASSES if t.name == tool_name]
        if not matching:
            result_str = f"Error: tool '{tool_name}' not found."
        else:
            result_str = matching[0]._run(**tool_args)

        # Vision result: forward image to VLM rather than passing raw base64
        from nodes.tool_use import _handle_image_result
        result_str = _handle_image_result(result_str, state.request.get("user_input", ""))

        _think(f"{tool_name} result", result_str)

    # ── Store result and advance ──────────────────────────────────────────────
    plan_results = dict(state.plan_results or {})
    plan_results[output_key] = result_str
    state.plan_results = plan_results
    state.plan_step = idx + 1

    state.history.append({
        "node":       "plan_step",
        "step":       step_num,
        "tool":       target_tool,
        "output_key": output_key,
        "result":     result_str,
    })
    return state


def plan_step_router(state: BoxState) -> str:
    """Continue executing steps, or finish when all are done."""
    if state.plan and state.plan_step < len(state.plan):
        return "continue"
    return "done"


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Summary — final natural-language answer
# ─────────────────────────────────────────────────────────────────────────────

def plan_summary_fn(state: BoxState) -> BoxState:
    """Synthesise all step results into a concise final answer."""
    results_text = "\n".join(
        f"  [{k}]: {v}" for k, v in (state.plan_results or {}).items()
    )
    prompt = (
        f"A multi-step Grasshopper design task has just been completed.\n\n"
        f"Original request: \"{state.request.get('user_input', '')}\"\n\n"
        f"Step results:\n{results_text}\n\n"
        "Write a short, clear summary for the user: what was created and any key values."
    )
    print(f"\n  ┊ synthesising plan summary...")
    try:
        resp = chat_llm._generate([HumanMessage(content=prompt)])
        state.answer = resp.generations[0].message.content
    except Exception:
        state.answer = (
            f"Plan completed in {len(state.plan or [])} steps.\n\n{results_text}"
        )
    state.done = True
    state.history.append({"node": "plan_summary", "answer": state.answer})
    return state
