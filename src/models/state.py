import operator
from pydantic import BaseModel, Field
from typing import Annotated, Dict, Any, List, Optional, Union, Literal

class BoxState(BaseModel):
    """State model for the architectural assistant agent."""
    # Basic request and response
    request: Dict[str, Any]
    answer: Optional[str] = None
    done: Optional[bool] = None
    
    # Input classification
    request_type: Optional[Literal["design_building", "show_guide", "general_question", "use_tool", "plan", "unknown"]] = None

    # GH tool execution results
    tool_results: Optional[Dict[str, Any]] = None

    # Building design
    box: Optional[Dict[str, Any]] = None
    compliant: Optional[bool] = None
    issues: Optional[List[str]] = None
    rules: Optional[List[Dict[str, Any]]] = None
    current_width: Optional[float] = None
    
    # ReAct components
    thought: Optional[str] = None
    action: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    
    # Search components
    search_query: Optional[str] = None
    search_results: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    needs_search: Optional[bool] = None
    
    # Additional parameters for expanded constraints
    window_area: Optional[float] = None
    emergency_exits: Optional[bool] = None
    aspect_ratio: Optional[float] = None
    
    # Plan execution (chained multi-tool tasks)
    plan: Optional[List[Dict[str, Any]]] = None          # [{step, tool, intent, output_key}, ...]
    plan_step: int = 0                                    # index of next step to execute
    plan_results: Dict[str, Any] = Field(default_factory=dict)  # {output_key: result_str}

    # Conversation memory – list of {"role": "user"|"assistant", "content": "..."}
    # Annotated with operator.add so LangGraph accumulates messages across turns
    # when a MemorySaver checkpointer is used (appended, not replaced).
    messages: Annotated[List[Dict[str, str]], operator.add] = Field(default_factory=list)

    # History tracking
    history: List[Dict[str, Any]] = Field(default_factory=list)