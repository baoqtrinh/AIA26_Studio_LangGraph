from typing import Dict, Any, List, Optional
from models.state import BoxState

def retrieve_rules_fn(state: BoxState) -> BoxState:
    """Retrieve the building code rules and constraints."""
    # Define building code rules
    rules = [
        {
            "id": "depth_constraint",
            "description": "Building depth must not exceed 50 meters for safety reasons",
        },
        {
            "id": "ratio_constraint",
            "description": "Width to depth ratio must be at least 0.33 (1:3) for structural stability",
        },
        {
            "id": "emergency_exit_constraint",
            "description": "Buildings must have emergency exits",
        }
    ]
    
    state.rules = rules
    return state

def thinking_fn(state: BoxState) -> BoxState:
    """ReAct thinking step that analyzes the current state and decides what to do next."""
    box = state.box or {}
    issues = state.issues or []
    
    # Generate a thought based on current state
    prompt = f"""
    You are an architectural assistant helping to design a building.
    
    Current design parameters:
    {box}
    
    Current issues:
    {issues}
    
    Previous observation:
    {state.observation}
    
    Think about how to adjust the design to meet all requirements.
    Focus on resolving issues while maintaining overall design integrity.
    """
    
    # Note: Replace this with your actual LLM implementation
    # Simulate LLM response for now
    thought = "I need to adjust the building parameters to comply with regulations."
    state.thought = thought
    
    # Add to history
    if state.history is None:
        state.history = []
    state.history.append({
        "node": "thinking",
        "thought": thought
    })
    
    return state

def action_fn(state: BoxState) -> BoxState:
    """Decide what action to take based on thinking."""
    import math
    issues = state.issues or []
    area = state.request.get("area", 800)

    # Compute the minimum width that satisfies both geometry constraints:
    #   depth = area / width ≤ 50  →  width ≥ area / 50
    #   ratio = width / depth ≥ 0.33  →  width² ≥ 0.33 * area  →  width ≥ sqrt(0.33 * area)
    # Take the larger of the two, add a small margin and round up.
    min_for_depth = area / 50.0
    min_for_ratio = math.sqrt(0.33 * area)
    required_width = math.ceil(max(min_for_depth, min_for_ratio)) + 1

    has_depth  = any("depth_constraint"          in i for i in issues)
    has_ratio  = any("ratio_constraint"          in i for i in issues)
    has_exits  = any("emergency_exit_constraint" in i for i in issues)

    if has_depth or has_ratio:
        # Set width large enough to resolve both at once
        state.action = {"action": "adjust_width", "params": {"width": required_width}}
    elif has_exits:
        state.action = {"action": "add_emergency_exits", "params": {"emergency_exits": True}}
    else:
        # No known issues — nudge width slightly and let compliance re-evaluate
        state.action = {"action": "adjust_width", "params": {"width": (state.current_width or 16) + 4}}
    
    # Add to history
    state.history.append({
        "node": "action",
        "action": state.action
    })
    
    return state

def draw_box_fn(state: BoxState) -> BoxState:
    """Execute the chosen action and update the box parameters."""
    if state.box is None:
        state.box = {}
    
    action = state.action
    
    # Default initialization for first run
    if not action or state.current_width is None:
        state.current_width = 10
        width = state.current_width
        area = state.request.get("area", 800)
        n_floors = state.request.get("n_floors", 2)
        floor_height = state.request.get("floor_height", 3)
    else:
        # Execute the action from the ReAct process
        action_type = action.get("action", "adjust_width")
        params = action.get("params", {})
        
        if action_type == "adjust_width":
            state.current_width = params.get("width", state.current_width + 2)
        elif action_type == "adjust_depth":
            state.box["depth"] = params.get("depth")
        elif action_type == "adjust_floors":
            state.box["n_floors"] = params.get("n_floors")
        elif action_type == "adjust_floor_height":
            state.box["floor_height"] = params.get("floor_height")
        elif action_type == "add_emergency_exits":
            state.emergency_exits = True
        elif action_type == "adjust_window_area":
            state.window_area = params.get("window_area")
        
        # Update current parameters
        width = state.current_width
        area = state.request.get("area", 800)
        n_floors = state.box.get("n_floors", state.request.get("n_floors", 2))
        floor_height = state.box.get("floor_height", state.request.get("floor_height", 3))
    
    # Calculate depth based on area and width
    depth = None
    if area is not None and width is not None:
        # Simplified calculation for now
        depth = area / width
    
    # Calculate additional parameters
    if width and depth:
        aspect_ratio = width / depth
    else:
        aspect_ratio = None
        
    if n_floors and floor_height:
        height = n_floors * floor_height
    else:
        height = None
    
    # Update the box state with all parameters
    state.box.update({
        "width": width,
        "depth": depth,
        "area": area,
        "n_floors": n_floors,
        "floor_height": floor_height,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "emergency_exits": state.emergency_exits,
        "window_area": state.window_area
    })
    
    # Add to history
    state.history.append({
        "node": "draw_box",
        "box": state.box.copy()
    })
    
    return state

def compliance_check_fn(state: BoxState) -> BoxState:
    """Check if the current box design meets all constraints."""
    box = state.box
    rules = state.rules or []
    
    # Check compliance against all rules
    _validators = {
        "depth_constraint": lambda b: b.get("depth", 0) <= 50,
        "ratio_constraint": lambda b: b.get("aspect_ratio", 0) >= 0.33,
        "emergency_exit_constraint": lambda b: b.get("emergency_exits", False) is True,
    }
    issues = []
    for rule in rules:
        try:
            validate = _validators.get(rule["id"])
            if validate and not validate(box):
                issues.append(f"Failed {rule['id']}: {rule['description']}")
        except Exception as e:
            issues.append(f"Error checking {rule['id']}: {str(e)}")
    
    # Update the state with compliance issues
    state.issues = issues
    
    # Set observation for the ReAct agent
    if issues:
        state.observation = f"Design does not comply with {len(issues)} rules: {', '.join(issues)}"
    else:
        state.observation = "Design complies with all rules."
    
    # Add to history
    state.history.append({
        "node": "compliance_check",
        "issues": issues.copy() if issues else []
    })
    
    return state

def is_compliant_fn(state: BoxState) -> BoxState:
    """Determine if the design is compliant based on issues."""
    # Design is compliant if there are no issues
    state.compliant = len(state.issues or []) == 0
    
    # Add to history
    state.history.append({
        "node": "is_compliant",
        "compliant": state.compliant
    })
    
    return state
