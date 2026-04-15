from langgraph.graph import StateGraph

from models.state import AgentState
from nodes.classification_node import classify_input_fn
from nodes.information_node import show_guide_fn, handle_unknown_fn
from nodes.deep_research_node import (
    deep_decompose_fn,
    deep_search_fn,
    deep_reflect_fn,
    deep_reflect_router,
    deep_followup_fn,
    deep_synthesize_fn,
)
from nodes.building_design_node import (
    building_planner_fn,
    retrieve_rules_fn,
    thinking_fn,
    action_fn,
    create_building_as_box,
    compliance_check_fn,
    is_compliant_fn,
)
from nodes.climate_optimization_node import (
    climate_investigate_fn,
    climate_replan_fn,
    execute_climate_step_fn,
    climate_step_router,
    climate_summary_fn,
)


def build_main_graph(checkpointer=None):
    """Build the LangGraph agent.

    Branches
    ────────
    design_building      → building_planner → retrieve_rules → thinking → execute_action
                         → create_building_as_box → compliance_check → is_compliant → END
                             Plan & Execute: plan committed upfront, ReAct loop executes it.

    climate_optimization → building_planner → [same ReAct compliance loop]
                         → climate_investigate → climate_replan → execute_climate_step (loop)
                         → climate_summary → END
                             Plan & Execute (outer) + Plan+Replan (inner climate stage).

    use_tool             → handle_unknown  (placeholder until a single-tool node exists)

    show_guide       → show_guide
    general_question → deep_decompose → deep_search → deep_reflect
                     → [gaps?] deep_followup → deep_synthesize
                                             → deep_synthesize
                         Deep Research: decomposes query, iterative multi-search, reflect, synthesize.
    unknown          → handle_unknown
    """
    g = StateGraph(AgentState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    g.add_node("classify_input",       classify_input_fn)

    # ── Branch A: Building design — Plan & Execute wrapper + ReAct loop ─────────
    g.add_node("building_planner",     building_planner_fn)
    g.add_node("retrieve_rules",       retrieve_rules_fn)
    g.add_node("thinking",             thinking_fn)
    g.add_node("execute_action",       action_fn)
    g.add_node("create_building_as_box", create_building_as_box)
    g.add_node("compliance_check",     compliance_check_fn)
    g.add_node("is_compliant",         is_compliant_fn)

    # ── Branch B2: Climate optimization Plan+Replan pipeline ──────────
    g.add_node("climate_investigate",  climate_investigate_fn)
    g.add_node("climate_replan",       climate_replan_fn)
    g.add_node("execute_climate_step", execute_climate_step_fn)
    g.add_node("climate_summary",      climate_summary_fn)

    # ── Branch C: Informational ────────────────────────────────────────
    g.add_node("show_guide",           show_guide_fn)
    g.add_node("handle_unknown",       handle_unknown_fn)

    # ── Branch D: Deep Research Agent ────────────────────────────────────
    g.add_node("deep_decompose",   deep_decompose_fn)
    g.add_node("deep_search",      deep_search_fn)
    g.add_node("deep_reflect",     deep_reflect_fn)
    g.add_node("deep_followup",    deep_followup_fn)
    g.add_node("deep_synthesize",  deep_synthesize_fn)

    # ── Entry + routing ────────────────────────────────────────────────────────
    g.set_entry_point("classify_input")

    g.add_conditional_edges(
        "classify_input",
        lambda state: state.request_type,
        {
            "climate_optimization": "building_planner",
            "design_building":      "building_planner",
            "use_tool":             "handle_unknown",
            "show_guide":        "show_guide",
            "general_question":  "deep_decompose",
            "unknown":           "handle_unknown",
        },
    )

    # Branch A: Plan & Execute — planner then ReAct compliance loop
    g.add_edge("building_planner", "retrieve_rules")
    g.add_edge("retrieve_rules",   "thinking")
    g.add_edge("thinking",         "execute_action")
    g.add_edge("execute_action",   "create_building_as_box")
    g.add_edge("create_building_as_box", "compliance_check")
    g.add_edge("compliance_check", "is_compliant")
    def _is_compliant_router(state: AgentState) -> str:
        if not state.context.get("compliant"):
            return "retry"
        if state.request_type == "climate_optimization":
            return "climate_optimize"
        return "done"

    g.add_conditional_edges(
        "is_compliant",
        _is_compliant_router,
        {"done": "__end__", "retry": "thinking", "climate_optimize": "climate_investigate"},
    )

    # Branch B2: Climate optimization Plan+Replan pipeline
    g.add_edge("climate_investigate", "climate_replan")
    g.add_edge("climate_replan",      "execute_climate_step")
    g.add_conditional_edges(
        "execute_climate_step",
        climate_step_router,
        {"continue": "execute_climate_step", "done": "climate_summary"},
    )
    g.add_edge("climate_summary", "__end__")

    # Branch C terminals
    g.add_edge("show_guide",     "__end__")
    g.add_edge("handle_unknown", "__end__")

    # Branch D: Deep Research Agent
    g.add_edge("deep_decompose", "deep_search")
    g.add_edge("deep_search",    "deep_reflect")
    g.add_conditional_edges(
        "deep_reflect",
        deep_reflect_router,
        {"follow_up": "deep_followup", "synthesize": "deep_synthesize"},
    )
    g.add_edge("deep_followup",   "deep_synthesize")
    g.add_edge("deep_synthesize", "__end__")

    return g.compile(checkpointer=checkpointer)
