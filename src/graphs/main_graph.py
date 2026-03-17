from langgraph.graph import StateGraph

from models.state import BoxState
from nodes.classification import classify_input_fn
from nodes.information import show_guide_fn, handle_unknown_fn
from nodes.search import (
    determine_search_need_fn,
    perform_web_search_fn,
    answer_with_search_fn,
    answer_without_search_fn,
)
from nodes.building_design import (
    retrieve_rules_fn,
    thinking_fn,
    action_fn,
    draw_box_fn,
    compliance_check_fn,
    is_compliant_fn,
)
from nodes.tool_use import execute_gh_tool_fn
from nodes.planning import (
    planner_fn,
    plan_step_fn,
    plan_step_router,
    plan_summary_fn,
)


def build_main_graph(checkpointer=None):
    """Build the LangGraph agent.

    Branches
    ────────
    plan             → planner → plan_step (loop) → plan_summary
                         Chain multiple GH tools in sequence (plan mode).

    use_tool         → execute_gh_tool
                         Draw / model a single geometry via GH script.

    design_building  → retrieve_rules → thinking → execute_action
                     → draw_box → compliance_check → is_compliant
                         Size a building against code constraints (ReAct loop).

    show_guide       → show_guide
    general_question → determine_search_need → [web_search |] → answer
    unknown          → handle_unknown
    """
    g = StateGraph(BoxState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    g.add_node("classify_input",       classify_input_fn)

    # ── Branch A: Plan mode (multi-tool chaining) ────────────────────────
    g.add_node("planner",              planner_fn)
    g.add_node("execute_plan_step",    plan_step_fn)
    g.add_node("plan_summary",         plan_summary_fn)

    # ── Branch B: Single GH geometry tool ────────────────────────────────
    g.add_node("execute_gh_tool",      execute_gh_tool_fn)

    # ── Branch B: Building design ReAct loop ───────────────────────────
    g.add_node("retrieve_rules",       retrieve_rules_fn)
    g.add_node("thinking",             thinking_fn)
    g.add_node("execute_action",       action_fn)
    g.add_node("draw_box",             draw_box_fn)
    g.add_node("compliance_check",     compliance_check_fn)
    g.add_node("is_compliant",         is_compliant_fn)

    # ── Branch C: Informational ────────────────────────────────────────
    g.add_node("show_guide",           show_guide_fn)
    g.add_node("handle_unknown",       handle_unknown_fn)

    # ── Branch D: General Q&A + optional web search ────────────────────
    g.add_node("determine_search_need",  determine_search_need_fn)
    g.add_node("perform_web_search",     perform_web_search_fn)
    g.add_node("answer_with_search",     answer_with_search_fn)
    g.add_node("answer_without_search",  answer_without_search_fn)

    # ── Entry + routing ────────────────────────────────────────────────────────
    g.set_entry_point("classify_input")

    g.add_conditional_edges(
        "classify_input",
        lambda state: state.request_type,
        {
            "plan":              "planner",
            "use_tool":          "execute_gh_tool",
            "design_building":   "retrieve_rules",
            "show_guide":        "show_guide",
            "general_question":  "determine_search_need",
            "unknown":           "handle_unknown",
        },
    )

    # Branch A: plan mode loop
    g.add_edge("planner", "execute_plan_step")
    g.add_conditional_edges(
        "execute_plan_step",
        plan_step_router,
        {"continue": "execute_plan_step", "done": "plan_summary"},
    )
    g.add_edge("plan_summary", "__end__")

    # Branch B terminal
    g.add_edge("execute_gh_tool", "__end__")

    # Branch C: ReAct loop
    g.add_edge("retrieve_rules",   "thinking")
    g.add_edge("thinking",         "execute_action")
    g.add_edge("execute_action",   "draw_box")
    g.add_edge("draw_box",         "compliance_check")
    g.add_edge("compliance_check", "is_compliant")
    g.add_conditional_edges(
        "is_compliant",
        lambda state: "True" if state.compliant else "False",
        {"True": "__end__", "False": "thinking"},
    )

    # Branch C terminals
    g.add_edge("show_guide",    "__end__")
    g.add_edge("handle_unknown","__end__")

    # Branch D
    g.add_conditional_edges(
        "determine_search_need",
        lambda state: "needs_search" if state.needs_search else "no_search",
        {"needs_search": "perform_web_search", "no_search": "answer_without_search"},
    )
    g.add_edge("perform_web_search",    "answer_with_search")
    g.add_edge("answer_with_search",    "__end__")
    g.add_edge("answer_without_search", "__end__")

    return g.compile(checkpointer=checkpointer)
