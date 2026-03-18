import operator
from pydantic import BaseModel, Field
from typing import Annotated, Dict, Any, List, Optional, Literal


class BoxState(BaseModel):
    """Agent state — slim universal core + one flexible workflow scratchpad.

    Core fields (always present, typed, used by the router or LLM):
        request      — raw input dict: {user_input, site_area, width, n_floors, …}
        request_type — branch the classifier chose
        answer       — final response shown to the user
        done         — True once the run is finished
        observation  — last ReAct observation (fed into the next thinking step)
        messages     — conversation history (LangGraph reducer: accumulates across turns)
        history      — ordered {node, …} audit trail for debugging

    context — workflow-specific scratchpad (Dict[str, Any]):
        Nodes read and write state.context["key"] freely.
        Adding a new tool or workflow never requires editing this file.

        Keys used today
        ───────────────
        Building design : box, rules, issues, compliant
                          current_width, current_n_floors
                          thought, action, llm_action
                          window_area, emergency_exits
        Search          : needs_search, search_query, search_results
        Plan & Execute  : plan, plan_step, plan_results
        Single tool     : tool_results
    """

    # ── Core (universal) ──────────────────────────────────────────────────────
    request:      Dict[str, Any]
    request_type: Optional[Literal[
        "design_building", "show_guide", "general_question",
        "use_tool", "plan", "unknown"
    ]] = None
    answer:      Optional[str]  = None
    done:        Optional[bool] = None
    observation: Optional[str]  = None   # last ReAct observe step

    # ── Workflow scratchpad ───────────────────────────────────────────────────
    context: Dict[str, Any] = Field(default_factory=dict)

    # ── Conversation memory (reducer: accumulates across LangGraph turns) ─────
    messages: Annotated[List[Dict[str, str]], operator.add] = Field(default_factory=list)

    # ── Debug / audit trail ───────────────────────────────────────────────────
    history: List[Dict[str, Any]] = Field(default_factory=list)