"""Climate optimization nodes — climate_optimization branch.

This branch demonstrates Plan & Execute + Replanning:

  Outer plan skeleton (written before execution starts):
    Step A : Design compliant building   [ReAct loop — building_design_node]
    Step B : Investigate local climate   [web-search for {location} climate]
    ── REPLAN ──────────────────────────────────────────────────────────────
    Step C+: Written by the LLM AFTER reading the raw search results.
             The LLM decides rotation angle and extra notes from evidence —
             not from pre-set rules.  Step count can vary per location.

Graph path
──────────
is_compliant (compliant=True, climate_optimization)
  → climate_investigate   (web-search + classify climate type)
  → climate_replan        (LLM rewrites remaining steps based on climate)
  → execute_climate_step  (loop — execute each replanned step)
  → climate_summary       (final report)
  → END
"""
import json
import re
import textwrap
import requests

from models.state import AgentState
from utils.llm_utils import fast_llm, llm
from tools.search.tools import search_web

try:
    from settings import MCP_GH_ENDPOINT, MCP_TIMEOUT
except ImportError:
    import os
    MCP_GH_ENDPOINT = os.getenv("MCP_GH_ENDPOINT", "http://localhost:5001/mcp/")
    MCP_TIMEOUT = int(os.getenv("MCP_TIMEOUT", "30"))

_HR = "─" * 72


def _think(label: str, text: str):
    prefix = f"  ┊ {label}: "
    body = str(text).strip().replace("\n", " ")
    for i, line in enumerate(textwrap.wrap(body, width=68)):
        print((prefix if i == 0 else " " * len(prefix)) + line)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Climate investigation — observation that drives the replan
# ─────────────────────────────────────────────────────────────────────────────

def climate_investigate_fn(state: AgentState) -> AgentState:
    """Web-search + classify climate type.

    This is the key observation step: the result (tropical / hot_arid /
    temperate / cold) determines which remaining steps the LLM will write
    in the next node (solar_replan).  We could NOT have written those steps
    before running this node — hence replanning is required.
    """
    user_input = state.request.get("user_input", "")

    print(f"\n{_HR}")
    print(f"  ▶  CLIMATE INVESTIGATE  —  classifying climate for location...")
    print(_HR)

    # 1) Extract location
    location_prompt = (
        f'Extract the city or country name from this architecture request.\n'
        f'Return ONLY the location name, nothing else.\n'
        f'If no location is mentioned, return "unknown".\n\n'
        f'Request: "{user_input}"\n\nLocation:'
    )
    location = str(fast_llm(location_prompt)).strip().strip('"\'')
    _think("location", location)
    state.context["climate_location"] = location

    if location.lower() == "unknown":
        state.context["climate_type"]    = "temperate"
        state.context["climate_summary"] = (
            "No location specified — defaulting to temperate. "
            "Will rotate toward the sun as a conservative assumption."
        )
        _think("climate_type", "temperate (default — no location)")
        state.history.append({
            "node": "climate_investigate", "location": location,
            "climate_type": "temperate",
        })
        return state

    # 2) Web search
    query = f"climate type {location} architecture solar design passive"
    _think("search query", query)
    try:
        raw = search_web.invoke(query)
        results = raw["results"] if isinstance(raw, dict) and "results" in raw else (
            raw if isinstance(raw, list) else []
        )
    except Exception as exc:
        results = []
        _think("search error", str(exc))

    state.context["climate_search_results"] = results
    _think("search hits", str(len(results)))

    snippets = "\n".join(
        f"- {r.get('title','')}: {(r.get('content') or r.get('snippet',''))[:300]}"
        for r in results[:5]
    )
    state.context["climate_search_snippets"] = snippets
    _think("snippets", snippets[:120] if snippets else "(none)")

    state.history.append({
        "node":        "climate_investigate",
        "location":    location,
        "search_hits": len(results),
    })
    return state


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Climate replan — LLM rewrites remaining steps based on climate finding
# ─────────────────────────────────────────────────────────────────────────────

def climate_replan_fn(state: AgentState) -> AgentState:
    """The replanning node.

    The LLM reads the raw web-search snippets found by climate_investigate
    and freely decides what orientation steps to execute — rotation angle,
    number of steps, and any design notes.  Because the decision comes from
    open-ended search results (not a fixed category lookup), the output
    genuinely varies per location.
    """
    ctx      = state.context
    location = ctx.get("climate_location", "unknown")
    snippets = ctx.get("climate_search_snippets", "")
    box      = ctx.get("box") or {}

    print(f"\n{_HR}")
    print(f"  ▶  CLIMATE REPLAN  —  reasoning from web evidence for {location}...")
    print(_HR)

    replan_prompt = f"""You are a solar design consultant optimising building orientation for a specific location.

Code-compliant building already created:
  Site area    : {box.get('site_area', '?')} m²
  Width        : {box.get('width', '?')} m
  Floors       : {box.get('n_floors', '?')}
  Floor height : {box.get('floor_height', '?')} m
  Location     : {location}

Web research findings about {location} climate and solar design:
{snippets if snippets else '(no search results — use your architectural knowledge)'}

Available actions:
  rotate_building  — rotate the building in Grasshopper
    params: {{"degree": <float 0-360>}}
    convention: 0 = main facade faces North
                90 = East · 180 = South · 270 = West
  add_design_note  — record a design recommendation (no GH call)
    params: {{"note": "<text>"}}

Your task:
  Based ONLY on the research above, decide:
  1. The optimal building orientation (which degree to rotate to) and why.
  2. Any additional design notes (shading, ventilation, insulation, etc.).

  Do NOT follow a fixed rule — reason from the evidence.
  The number of steps is up to you (minimum 1, typically 1–3).

Write the remaining steps as a JSON array. Each item:
  "step"   : integer starting at 1
  "action" : "rotate_building" | "add_design_note"
  "params" : object with the required fields
  "note"   : one sentence citing what in the research led to this decision

Respond ONLY with the JSON array (no markdown):"""

    try:
        raw   = str(fast_llm(replan_prompt)).strip()
        raw   = re.sub(r"^```(?:json)?\s*", "", raw)
        raw   = re.sub(r"\s*```$", "", raw)
        steps = json.loads(raw)
        assert isinstance(steps, list) and len(steps) > 0
    except Exception:
        # Minimal fallback — face south (safe default for most locations)
        steps = [
            {"step": 1, "action": "rotate_building",
             "params": {"degree": 180.0},
             "note": "Default: face south to maximise solar exposure (fallback — LLM parse failed)."},
        ]

    ctx["climate_remaining_steps"] = steps
    ctx["climate_step_idx"]        = 0
    ctx["climate_step_results"]    = []

    print(f"  ┊ {len(steps)}-step replan for {location}:")
    for s in steps:
        print(f"  ┊   [{s.get('step','?')}] {s.get('action','?')}  —  {s.get('note','')}")

    state.history.append({
        "node":    "climate_replan",
        "location": location,
        "n_steps": len(steps),
        "steps":   steps,
    })
    return state


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Execute climate step — loop over the replanned steps
# ─────────────────────────────────────────────────────────────────────────────

def execute_climate_step_fn(state: AgentState) -> AgentState:
    """Execute the current step from the LLM-generated replan list."""
    ctx   = state.context
    steps = ctx.get("climate_remaining_steps") or []
    idx   = ctx.get("climate_step_idx", 0)

    if idx >= len(steps):
        return state

    step     = steps[idx]
    step_num = step.get("step", idx + 1)
    action   = step.get("action", "")
    params   = step.get("params") or {}
    total    = len(steps)

    print(f"\n  ── Climate step {step_num}/{total}: [{action}]  —  {step.get('note', '')}")

    result = ""

    if action == "rotate_building":
        degree  = float(params.get("degree", 180.0))
        payload = {
            "jsonrpc": "2.0", "id": 20 + idx,
            "method": "tools/call",
            "params": {
                "name": "rotate_building_toward_sunny_direction",
                "arguments": {"degree": degree},
            },
        }
        try:
            resp = requests.post(
                MCP_GH_ENDPOINT, json=payload,
                headers={"Content-Type": "application/json"},
                timeout=MCP_TIMEOUT,
            )
            resp.raise_for_status()
            data    = resp.json()
            success = not bool(data.get("error"))
            result  = str(data.get("result", {}).get("content", "")) if success else f"MCP error: {data.get('error')}"
        except Exception as exc:
            success = False
            result  = f"MCP call failed: {exc}"

        ctx["rotate_result"]   = result
        ctx["rotate_success"] = success
        ctx["climate_azimuth"] = degree
        _think(f"rotate({degree}°)", result[:80] or "ok")

    elif action == "add_design_note":
        note_text = params.get("note", step.get("note", ""))
        result    = note_text
        notes     = ctx.get("design_notes") or []
        notes.append(note_text)
        ctx["design_notes"] = notes
        _think("design note", note_text)

    else:
        result = f"Unknown action: {action}"
        _think("warning", result)

    step_results = ctx.get("climate_step_results") or []
    step_results.append({"step": step_num, "action": action, "result": str(result)[:200]})
    ctx["climate_step_results"] = step_results
    ctx["climate_step_idx"]     = idx + 1

    state.history.append({
        "node":   "execute_climate_step",
        "step":   step_num,
        "action": action,
        "result": str(result)[:200],
    })
    return state


def climate_step_router(state: AgentState) -> str:
    """'continue' while steps remain in the replan list, 'done' when finished."""
    steps = state.context.get("climate_remaining_steps") or []
    idx   = state.context.get("climate_step_idx", 0)
    return "continue" if idx < len(steps) else "done"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Climate summary — compile final answer
# ─────────────────────────────────────────────────────────────────────────────

def climate_summary_fn(state: AgentState) -> AgentState:
    """Produce a final natural-language summary of the climate optimization workflow."""
    ctx = state.context
    box = ctx.get("box") or {}

    steps_text = "\n".join(
        f"  Step {r['step']} ({r['action']}): {r['result'][:100]}"
        for r in (ctx.get("climate_step_results") or [])
    )
    notes_text = "\n".join(
        f"  - {n}" for n in (ctx.get("design_notes") or [])
    ) or "  (none)"

    prompt = f"""You are an architecture project assistant.
Summarise this climate-optimised building design in 3–5 sentences for the user.

Code-compliant building:
  Site area: {box.get('site_area', '?')} m²  ·  Width: {box.get('width', '?')} m
  Floors: {box.get('n_floors', '?')}  ·  Floor height: {box.get('floor_height', '?')} m

Location        : {ctx.get('climate_location', 'unknown')}
Search evidence : {str(ctx.get('climate_search_snippets', ''))[:300]}

Replanned steps executed:
{steps_text}

Design recommendations:
{notes_text}

Mention: (1) what the research revealed about the local climate, (2) whether
the building faces toward or away from the sun and why, (3) the rotation
degree applied, (4) any design notes."""

    try:
        summary = str(llm(prompt)).strip()
    except Exception:
        summary = (
            f"Building designed for {ctx.get('climate_location', '?')} "
            f"({ctx.get('climate_type', '?')} climate): "
            f"{box.get('n_floors', '?')} floor(s), GFA = {box.get('gfa', '?')} m². "
            f"{ctx.get('climate_summary', '')}"
        )

    state.answer = summary
    _think("summary", summary[:120])
    state.history.append({"node": "climate_summary", "answer": summary})
    return state


_HR = "─" * 72


