from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph
from langchain_core.tools import tool
import requests
import re
import json

load_dotenv()
API_KEY = os.getenv("CLOUDFLARE_API_KEY")
ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")

class CloudflareLLM:
    def __init__(self, api_key, account_id, model="@cf/meta/llama-3.3-70b-instruct-fp8-fast"):
        self.api_key = api_key
        self.account_id = account_id
        self.model = model

    def __call__(self, prompt, **kwargs):
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "messages": [
                {"role": "system", "content": "You are a building code compliance expert. "
                "Given building design proposals and code requirements, "
                "determine if the proposal is compliant. "
                "Explain your reasoning step by step."},
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        # Return the full result for debugging
        return result

def extract_building_params_with_cf_llm(user_message, account_id):
    """
    Use Cloudflare LLM to extract building parameters and show reasoning.
    Returns a dict with width, depth, n_floors, floor_height, area (None if missing), and reasoning.
    """
    prompt = (
        "You are a helpful assistant. "
        "Given the user's message, first explain your reasoning step by step. "
        "Then, extract the following building parameters and return them as a JSON object: "
        "width (number or null), depth (number or null), n_floors (integer or null), floor_height (number or null), area (number or null). "
        "If a field is missing, set it to null.\n"
        "Example output:\n"
        "Reasoning:\n"
        "The user provided area, number of floors, and floor height, but did not specify width or depth. Therefore, width and depth are set to null.\n"
        "Extracted JSON:\n"
        "{\n"
        "  \"width\": null,\n"
        "  \"depth\": null,\n"
        "  \"n_floors\": 3,\n"
        "  \"floor_height\": 3.5,\n"
        "  \"area\": 800\n"
        "}\n"
    f"User message: {user_message}\n"
    "Reasoning:\n"
    "Extracted JSON:"
)
    llm = CloudflareLLM(API_KEY, account_id)
    llm_result = llm(prompt)
    # Extract the text output from the full result
    if isinstance(llm_result, dict) and "result" in llm_result and "response" in llm_result["result"]:
        response = llm_result["result"]["response"]
    else:
        response = str(llm_result)

    # Try to extract JSON from the response
    match = re.search(r'\{.*\}', response, re.DOTALL)
    json_obj = None
    if match:
        try:
            json_obj = json.loads(match.group(0))
        except Exception:
            json_obj = None

    # Always set llm_reasoning to the full response text
    return {
        "params": json_obj,
        "llm_reasoning": response,
    }

llm = CloudflareLLM(API_KEY, ACCOUNT_ID)

DESIGN_GUIDE = [
    {"rule": "Maximum area is 3000 sqm", "type": "area", "max": 3000},
    {"rule": "Maximum building height is 18m", "type": "height", "max": 18},
    {"rule": "Maximum building width is 20m", "type": "width", "max": 20},
    {"rule": "Maximum building depth is 50m", "type": "depth", "max": 50},
    {"rule": "Maximum number of floors is 4", "type": "n_floors", "max": 4},
]


@tool
def compute_other_dimension(area: float, known_dimension: float) -> float:
    """
    Given area and one dimension (width or depth), compute the other dimension.
    """
    if known_dimension == 0:
        raise ValueError("Known dimension cannot be zero.")
    return area / known_dimension

class BoxState(BaseModel):
    request: Dict[str, Any]
    box: Optional[Dict[str, Any]] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)
    compliant: Optional[bool] = None
    issues: Optional[List[str]] = None
    rules: Optional[List[Dict[str, Any]]] = None
    current_width: Optional[float] = None
    done: Optional[bool] = None

def draw_box_fn(state):
    area = state.request.get("area")
    n_floors = state.request.get("n_floors")
    floor_height = state.request.get("floor_height")
    if state.current_width is None:
        state.current_width = 10
    else:
        state.current_width += 5
    width = state.current_width
    depth = None
    if area is not None and width is not None:
        print(f"Trying to compute depth with area={area}, width={width}")
        try:
            depth = compute_other_dimension.invoke({"area": area, "known_dimension": width})
            print(f"Computed depth: {depth}")
        except Exception as e:
            print(f"Error computing depth: {e}")
            depth = None
    box = {
        "width": width,
        "depth": depth,
        "n_floors": n_floors,
        "floor_height": floor_height,
        "area": area,
        "height": n_floors * floor_height if n_floors and floor_height else None
    }
    state.box = box
    if state.history is None:
        state.history = []
    state.history.append({
        "node": "draw_box",
        "box": box.copy()
    })
    return state

def retrieve_rules_fn(state):
    state.rules = DESIGN_GUIDE
    state.history.append({"node": "retrieve_rules", "rules": DESIGN_GUIDE})
    return state

def compliance_check_fn(state):
    box = state.box
    rules = state.rules
    issues = []
    for dim in ["width", "depth", "n_floors", "floor_height", "area"]:
        if box.get(dim) is None:
            issues.append(f"{dim} is missing")
    if not issues:
        for rule in rules:
            if rule["type"] in box and box[rule["type"]] is not None:
                if box[rule["type"]] > rule["max"]:
                    issues.append(f"{rule['type']} exceeds max {rule['max']}")
    state.compliant = len(issues) == 0
    state.issues = issues
    if state.history is None:
        state.history = []
    state.history.append({"node": "compliance_check", "compliant": state.compliant, "issues": issues.copy()})
    print(f"Compliance check: {state.compliant}, issues: {issues}")  # Debug print
    return state

def is_compliant_fn(state):
    if state.compliant:
        state.done = True
        state.history.append({"node": "is_compliant", "result": "compliant"})
    else:
        state.done = False
        state.history.append({"node": "is_compliant", "result": "non-compliant"})
    return state  # Always return the state object

def build_graph():
    g = StateGraph(state_schema=BoxState)
    g.add_node("draw_box", draw_box_fn)
    g.add_node("retrieve_rules", retrieve_rules_fn)
    g.add_node("compliance_check", compliance_check_fn)
    g.add_node("is_compliant", is_compliant_fn)
    g.set_entry_point("draw_box")
    g.add_edge("draw_box", "retrieve_rules")
    g.add_edge("retrieve_rules", "compliance_check")
    g.add_edge("compliance_check", "is_compliant")
    g.add_conditional_edges(
        "is_compliant",
        lambda state: "True" if state.compliant else "False",
        {
            "True": "__end__",
            "False": "draw_box"
        }
    )
    compiled = g.compile()
    
    # Create a directory for saving the graph if it doesn't exist
    graph_dir = os.path.join(os.path.dirname(__file__), "graphs")
    os.makedirs(graph_dir, exist_ok=True)
    
    # Save the graph to the graphs directory
    graph_path = os.path.join(graph_dir, "design_agent_graph.png")
    with open(graph_path, "wb") as f:
        f.write(compiled.get_graph().draw_mermaid_png())
    
    print(f"Workflow graph saved as {graph_path}")
    return compiled

graph = build_graph()

if __name__ == "__main__":
    user_message = "I want a building with area 800, 3 floors, each 3.5m high"
    extraction = extract_building_params_with_cf_llm(user_message, ACCOUNT_ID)
    params = extraction.get("params") if extraction else None
    llm_reasoning = extraction.get("llm_reasoning") if extraction else None

    print("LLM Reasoning:\n" + (llm_reasoning or "No reasoning found.") + "\n" + "="*40)

    if not params:
        print("Could not extract parameters from your message.")
        exit()

    state = BoxState(request=params)
    
    # Use invoke() directly to get the final state
    final_state_dict = graph.invoke(state)
    
    # Convert the result back to your state model
    final_state = BoxState(**final_state_dict)
    
    print("\nFinal result:")
    print({
        "compliant": final_state.compliant,
        "issues": final_state.issues,
        "final_box": final_state.box,
        "history": final_state.history[-1] if final_state.history else None,
    })