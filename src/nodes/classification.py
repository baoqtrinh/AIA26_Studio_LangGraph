from models.state import BoxState
from utils.llm_utils import fast_llm

_HR = "─" * 72


def _think(label: str, text: str):
    """Print a dim thinking line to the terminal."""
    import textwrap
    prefix = f"  ┊ {label}: "
    body = text.strip().replace("\n", " ")
    for i, line in enumerate(textwrap.wrap(body, width=68)):
        print((prefix if i == 0 else " " * len(prefix)) + line)


def classify_input_fn(state: BoxState) -> BoxState:
    """Classify the user input into one of five routing categories."""
    user_input = state.request.get("user_input", "")

    # If request_type was pre-set (e.g. plan mode bypasses classifier), honour it
    if state.request_type is not None:
        _think("pre-set", state.request_type)
        return state

    if not user_input:
        state.request_type = "use_tool"
        return state

    # ── Dynamic tool section ──────────────────────────────────────────────────
    try:
        from tools import TOOL_CLASSES
        if TOOL_CLASSES:
            tool_names = ", ".join(t.name for t in TOOL_CLASSES)
            tool_section = (
                f"2. use_tool: The user wants to draw, create, generate or model "
                f"a specific 3-D shape using one of these Grasshopper tools: {tool_names}"
            )
        else:
            tool_section = (
                "2. use_tool: The user wants to draw, create, generate or model "
                "a specific 3-D shape (cylinder, box, wall, slab, curve …)"
            )
    except Exception:
        tool_section = (
            "2. use_tool: The user wants to draw, create or generate a specific "
            "3-D shape or run a Grasshopper tool"
        )

    prompt = f"""You are a routing assistant. Classify the user request into EXACTLY ONE category.

User request: "{user_input}"

Categories:
1. design_building: Design, size or check code compliance of a BUILDING as a whole
   (floors, total area, depth, structural ratios, emergency exits, building code).
{tool_section}
3. show_guide: Show design guidelines, rules or constraints
4. general_question: General architecture question that does NOT involve drawing or sizing a building
5. unknown: Does not fit any category above

Rules:
- Whole-building sizing with code compliance → design_building
- Drawing / modelling any specific geometry shape or running a named tool → use_tool
- Output ONLY the category name, nothing else.

Classification:"""

    try:
        response = fast_llm(prompt)
    except Exception as exc:
        print(f"  ┊ LLM error: {exc}")
        print(f"  ⇒ classified as: unknown (LLM unreachable)")
        state.request_type = "unknown"
        state.history.append({"node": "classify_input", "request_type": "unknown", "user_input": user_input})
        return state

    classification = str(response).strip().lower()
    _think("LLM raw", classification)

    if "design_building" in classification:
        state.request_type = "design_building"
    elif "use_tool" in classification:
        state.request_type = "use_tool"
    elif "show_guide" in classification:
        state.request_type = "show_guide"
    elif "general_question" in classification:
        state.request_type = "general_question"
    else:
        state.request_type = "unknown"

    print(f"  ⇒ classified as: {state.request_type}")
    print()
    state.history.append({
        "node": "classify_input",
        "request_type": state.request_type,
        "user_input": user_input,
    })
    return state
