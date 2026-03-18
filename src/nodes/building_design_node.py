import requests
from typing import Dict, Any, List, Optional
from models.state import BoxState

try:
    from settings import MCP_GH_ENDPOINT, MCP_TIMEOUT
except ImportError:
    import os
    MCP_GH_ENDPOINT = os.getenv("MCP_GH_ENDPOINT", "http://localhost:5001/mcp/")
    MCP_TIMEOUT = int(os.getenv("MCP_TIMEOUT", "30"))


def _call_mcp_create_building(area: float, width: float, number_of_floors: int, floor_height: float) -> str:
    """Forward create_building to the Grasshopper MCP server via JSON-RPC 2.0."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "create_building",
            "arguments": {
                "site_area": area,
                "width": width,
                "number_of_floors": int(number_of_floors),
                "floor_height": floor_height,
            },
        },
    }
    try:
        resp = requests.post(
            MCP_GH_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=MCP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return f"MCP error: {data['error']}"
        result = data.get("result", {})
        return str(result.get("content", result))
    except requests.exceptions.Timeout:
        return f"MCP timeout after {MCP_TIMEOUT}s — is Grasshopper running?"
    except requests.exceptions.ConnectionError:
        return "MCP connection error: is Grasshopper running?"
    except Exception as exc:
        return f"MCP call failed: {exc}"

def _call_mcp_add_emergency_exits() -> tuple[bool, str]:
    """Call the Grasshopper add_emergency_exits MCP tool. Returns (success, raw_result)."""
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "add_emergency_exits",
            "arguments": {"add": True},
        },
    }
    try:
        resp = requests.post(
            MCP_GH_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=MCP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error") is not None:
            return False, f"MCP error: {data['error']}"
        result = data.get("result", {})
        return True, str(result.get("content", result))
    except requests.exceptions.Timeout:
        return False, f"MCP timeout after {MCP_TIMEOUT}s — is Grasshopper running?"
    except requests.exceptions.ConnectionError:
        return False, "MCP connection error: is Grasshopper running?"
    except Exception as exc:
        return False, f"MCP call failed: {exc}"


def retrieve_rules_fn(state: BoxState) -> BoxState:
    """Load rules from DESIGN_GUIDE and auto-generate each rule's id as
    '{type}_constraint'.  Any new entry added to design_rules.py is picked
    up automatically — no mapping table needed here.
    """
    from config.design_rules import DESIGN_GUIDE

    rules = []
    for entry in DESIGN_GUIDE:
        rule_type = entry.get("type")
        if not rule_type:
            continue
        rules.append({
            "id":          f"{rule_type}_constraint",
            "description": entry["rule"],
            **{k: v for k, v in entry.items() if k in ("min", "max", "condition")},
        })

    state.context["rules"] = rules
    return state

def thinking_fn(state: BoxState) -> BoxState:
    """ReAct thinking step: call the LLM to reason about the current state,
    understand the user's intent (including inferring missing parameters), and
    decide the next action to take.  The LLM output is a JSON object:

        {
          "thought": "...",
          "action":  "adjust_width | adjust_floors | add_emergency_exits | done",
          "params":  { ... }   // action-specific, may be empty
        }
    """
    import json as _json
    from utils.llm_utils import fast_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    ctx    = state.context
    box    = ctx.get("box") or {}
    issues = ctx.get("issues") or []
    rules  = ctx.get("rules") or []
    req    = state.request or {}

    # Build a human-readable summary of the active rules for the LLM
    rules_text = "\n".join(
        f"  - {r['id']}: {r['description']}"
        + (f"  [max={r['max']}]" if "max" in r else "")
        + (f"  [min={r['min']}]" if "min" in r else "")
        for r in rules
    )

    system = (
        "You are an architectural design assistant operating inside a ReAct compliance loop.\n"
        "Your job: reason about the current building state and decide ONE corrective action.\n\n"
        "Available actions:\n"
        "  adjust_width   → params: {width: float}   — change the building footprint width\n"
        "  adjust_floors  → params: {n_floors: int}  — change the number of floors\n"
        "  add_emergency_exits → params: {}           — trigger the emergency-exit tool\n"
        "  done           → params: {}               — all rules satisfied, nothing to fix\n\n"
        "Rules in effect:\n"
        f"{rules_text}\n\n"
        "How GFA is calculated:\n"
        "  GFA = site_area × n_floors\n"
        "  Example: site_area=150, n_floors=25 → GFA=3750 (FAILS max 3000)\n"
        "  To fix: n_floors = floor(max_gfa / site_area) = floor(3000/150) = 20\n"
        "  Then also check n_floors against the n_floors_constraint max.\n\n"
        "Fix priority order (one issue per loop):\n"
        "  1. gfa_constraint or n_floors_constraint → use adjust_floors\n"
        "  2. width_constraint                      → use adjust_width\n"
        "  3. depth_constraint or ratio_constraint  → use adjust_width\n"
        "  4. emergency_exits_constraint            → use add_emergency_exits\n"
        "  5. no issues remain                      → use done\n\n"
        "When the user specifies a parameter (e.g. width=30), honour it unless it violates\n"
        "a rule — then clamp to the rule limit.\n\n"
        "Respond ONLY with valid JSON. No markdown, no extra text. Schema:\n"
        '{"thought": "...", "action": "<one of the actions above>", "params": {...}}'
    )

    human = (
        f"User request: {req}\n\n"
        f"Current box: {box if box else 'not drawn yet'}\n\n"
        f"Compliance issues: {issues if issues else 'none'}\n\n"
        f"Previous observation: {state.observation or 'none'}\n\n"
        "What is your thought and next action?"
    )

    try:
        response = fast_llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        decision = _json.loads(raw)
        state.context["thought"]    = decision.get("thought", "Analysing compliance issues.")
        state.context["llm_action"] = {
            "action": decision.get("action"),
            "params": decision.get("params") or {},
        }
    except Exception as exc:
        # LLM failed — clear llm_action so action_fn's rule-based fallback takes over
        print(f"  [thinking] LLM error ({type(exc).__name__}): {exc}")
        state.context["thought"]    = "LLM unavailable — using rule-based fallback."
        state.context["llm_action"] = None

    state.context["react_iteration"] = state.context.get("react_iteration", 0) + 1

    if state.history is None:
        state.history = []
    state.history.append({"node": "thinking", "thought": state.context["thought"]})
    return state


def action_fn(state: BoxState) -> BoxState:
    """Parse the LLM's action decision produced by thinking_fn and set state.action.
    Falls back to rule-based logic only if thinking_fn did not produce a valid action
    (e.g. during unit tests or when the LLM call failed).
    """
    import math

    ctx        = state.context
    llm_action = ctx.get("llm_action")

    if llm_action and llm_action.get("action") == "done":
        # LLM decided the design is finished — propagate "done" without any adjustment
        ctx["action"] = {"action": "done", "params": {}}
    elif llm_action and llm_action.get("action") not in (None, ""):
        # LLM produced a valid action — use it directly
        ctx["action"] = {"action": llm_action["action"], "params": llm_action.get("params", {})}
    else:
        # Fallback: rule-based decision (used in tests / when LLM is unavailable)
        issues    = ctx.get("issues") or []
        rules     = {r["id"]: r for r in (ctx.get("rules") or [])}
        site_area = state.request.get("site_area", state.request.get("area", 800))
        max_gfa      = rules.get("gfa_constraint",      {}).get("max", 3000)
        max_width    = rules.get("width_constraint",    {}).get("max", 20)
        max_n_floors = rules.get("n_floors_constraint", {}).get("max", 100)

        has_gfa      = any("gfa_constraint"             in i for i in issues)
        has_n_floors = any("n_floors_constraint"        in i for i in issues)
        has_width    = any("width_constraint"           in i for i in issues)
        has_depth    = any("depth_constraint"           in i for i in issues)
        has_ratio    = any("ratio_constraint"           in i for i in issues)
        has_exits    = any("emergency_exits_constraint" in i for i in issues)

        if has_gfa or has_n_floors:
            # Cap by both GFA limit and n_floors limit
            target = max(1, min(int(max_gfa / site_area), max_n_floors))
            ctx["action"] = {"action": "adjust_floors", "params": {"n_floors": target}}
        elif has_width:
            ctx["action"] = {"action": "adjust_width", "params": {"width": max_width}}
        elif has_depth or has_ratio:
            min_w = math.ceil(max(site_area / 50.0, math.sqrt(0.33 * site_area))) + 1
            ctx["action"] = {"action": "adjust_width", "params": {"width": min_w}}
        elif has_exits:
            ctx["action"] = {"action": "add_emergency_exits", "params": {}}
        elif not ctx.get("box"):
            req_width = state.request.get("width")
            if req_width:
                start_width = float(req_width)
            else:
                start_width = math.ceil(max(site_area / 50.0, math.sqrt(0.33 * site_area))) + 1
            ctx["action"] = {"action": "adjust_width", "params": {"width": start_width}}
        else:
            ctx["action"] = {"action": "adjust_width",
                             "params": {"width": (ctx.get("current_width") or 16) + 4}}

    state.history.append({"node": "action", "action": ctx.get("action")})
    return state

def create_building_as_box(state: BoxState) -> BoxState:
    """Execute the chosen action and update the box parameters."""
    ctx = state.context
    if ctx.get("box") is None:
        ctx["box"] = {}

    action = ctx.get("action")

    site_area    = state.request.get("site_area", state.request.get("area", 800))
    # current_n_floors persists across loop iterations; fall back to the request
    # value so stale checkpoints never override a new request.
    n_floors     = ctx.get("current_n_floors") if ctx.get("current_n_floors") is not None else state.request.get("n_floors", 2)
    floor_height = ctx["box"].get("floor_height", state.request.get("floor_height", 3))

    # Apply action (or initialise width on very first call)
    if action and action.get("action") == "done":
        # LLM says design is complete — skip redraw, let compliance_check verify current box
        state.history.append({"node": "create_building_as_box", "box": ctx["box"].copy()})
        return state

    if action:
        action_type = action.get("action", "adjust_width")
        params      = action.get("params", {})
        if action_type == "adjust_width":
            ctx["current_width"] = params.get("width", (ctx.get("current_width") or 10) + 2)
        elif action_type == "adjust_depth":
            ctx["box"]["depth"] = params.get("depth")
        elif action_type == "adjust_floors":
            n_floors = params.get("n_floors", n_floors)
            ctx["current_n_floors"] = n_floors
        elif action_type == "adjust_floor_height":
            floor_height = params.get("floor_height", floor_height)
        elif action_type == "add_emergency_exits":
            print("  [add_emergency_exits] → MCP add=True")
            ok, mcp_result = _call_mcp_add_emergency_exits()
            print(f"  [add_emergency_exits] ← {mcp_result}")
            ctx["emergency_exits"] = True
        elif action_type == "adjust_window_area":
            ctx["window_area"] = params.get("window_area")
    else:
        # Very first call — no action yet, use a sensible starting width
        if ctx.get("current_width") is None:
            ctx["current_width"] = max(10, int((site_area / 50) ** 0.5 * 1.5))

    width = ctx.get("current_width")
    gfa   = site_area * n_floors if (site_area and n_floors) else None

    # ── Call Grasshopper to draw the geometry ────────────────────────────────
    depth        = site_area / width if (site_area and width) else None
    aspect_ratio = width / depth if (width and depth) else None
    height       = n_floors * floor_height if (n_floors and floor_height) else None
    mcp_result   = None

    if width and n_floors and floor_height:
        print(f"  [create_building] → MCP site_area={site_area}, width={width}, n_floors={n_floors}, floor_height={floor_height}")
        mcp_raw = _call_mcp_create_building(
            area=float(site_area),
            width=float(width),
            number_of_floors=int(n_floors),
            floor_height=float(floor_height),
        )
        print(f"  [create_building] ← {mcp_raw}")
        mcp_result = mcp_raw

        # Parse GH response and prefer GH-computed values (more accurate)
        import json as _json, ast as _ast
        try:
            # mcp_raw is typically: "[{'type': 'text', 'text': '{...}'}]"
            items = _ast.literal_eval(mcp_raw) if isinstance(mcp_raw, str) else mcp_raw
            if isinstance(items, list):
                for item in items:
                    text = item.get("text", "{}") if isinstance(item, dict) else "{}"
                    try:
                        gh = _json.loads(text)
                    except Exception:
                        gh = {}
                    if gh:
                        # Use `or` fallback so locally-computed values survive when GH returns null
                        depth        = gh.get("depth") or depth
                        height       = gh.get("height") or height
                        # GH returns gross_floor_area and ratio_constraint
                        gfa          = gh.get("gross_floor_area") or gh.get("gfa") or gfa
                        _ratio_gh    = gh.get("ratio_constraint") or gh.get("ratio") or gh.get("aspect_ratio")
                        aspect_ratio = _ratio_gh if _ratio_gh is not None else (width / depth if (width and depth) else aspect_ratio)
                        break
        except Exception:
            pass  # keep locally computed fallbacks

    # Update the box state
    ctx["box"].update({
        "width": width,
        "depth": depth,
        "site_area": site_area,
        "gfa": gfa,
        "n_floors": n_floors,
        "floor_height": floor_height,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "emergency_exits": ctx.get("emergency_exits"),
        "window_area": ctx.get("window_area"),
        "mcp_result": mcp_result,
    })

    # Add to history
    state.history.append({
        "node": "create_building_as_box",
        "box": ctx["box"].copy()
    })

    return state

def compliance_check_fn(state: BoxState) -> BoxState:
    """Check if the current box design meets all constraints."""
    ctx   = state.context
    box   = ctx.get("box") or {}
    rules = ctx.get("rules") or []

    # Derive the box field name from the rule id: strip "_constraint" suffix.
    # The only non-obvious override is ratio → aspect_ratio (GH field name).
    _field_override = {"ratio_constraint": "aspect_ratio"}

    issues = []
    for rule in rules:
        rid   = rule["id"]
        field = _field_override.get(rid, rid.replace("_constraint", ""))
        try:
            val = box.get(field)
            if "condition" in rule:
                if not val:
                    issues.append(f"Failed {rid}: {rule['description']}")
            else:
                val = val or 0
                if "max" in rule and val > rule["max"]:
                    issues.append(f"Failed {rid}: {rule['description']}")
                if "min" in rule and val < rule["min"]:
                    issues.append(f"Failed {rid}: {rule['description']}")
        except Exception as e:
            issues.append(f"Error checking {rid}: {str(e)}")

    ctx["issues"] = issues

    if issues:
        state.observation = f"Design does not comply with {len(issues)} rules: {', '.join(issues)}"
    else:
        state.observation = "Design complies with all rules."

    state.history.append({
        "node": "compliance_check",
        "issues": issues.copy() if issues else []
    })

    return state

def is_compliant_fn(state: BoxState) -> BoxState:
    """Determine if the design is compliant based on issues."""
    issues    = state.context.get("issues") or []
    compliant = len(issues) == 0

    # Safety escape: if the agent has been looping too long, accept best-effort result
    max_iter = 8
    iteration = state.context.get("react_iteration", 0)
    if not compliant and iteration >= max_iter:
        remaining = ", ".join(issues)
        state.answer = (
            f"Could not fully satisfy all constraints after {iteration} iterations. "
            f"Remaining issues: {remaining}. "
            f"Best-effort design: {state.context.get('box') or {}}"
        )
        compliant = True  # force exit

    state.context["compliant"] = compliant

    state.history.append({
        "node": "is_compliant",
        "compliant": compliant
    })

    return state
