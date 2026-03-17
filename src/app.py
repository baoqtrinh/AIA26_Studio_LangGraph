from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as FastAPIModel
from typing import Dict, Any, Optional, List
import uvicorn
import os
from dotenv import load_dotenv

# Import the state model and graph
from models.state import BoxState
from graphs.main_graph import build_main_graph

# Create the graph
graph = build_main_graph()

# Create FastAPI app
app = FastAPI(
    title="Architectural Assistant API",
    description="API for interacting with an architectural assistant agent",
    version="1.0.0"
)

# Define request and response models
class ChatRequest(FastAPIModel):
    message: str
    history: Optional[List[Dict[str, Any]]] = []

class ChatResponse(FastAPIModel):
    response: str
    type: str  # "design", "guide", "answer", or "unknown"
    data: Optional[Dict[str, Any]] = None  # For design results or search results

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Process a chat message with the architectural assistant."""
    try:
        # Create state from the message
        if not request.message:
            return {"response": "Please provide a message.", "type": "error"}
        
        # Create initial state with user input
        state = BoxState(request={"user_input": request.message})
        
        # Run the agent
        final_state_dict = graph.invoke(state, config={"recursion_limit": 50})
        
        # Process the response based on request type
        request_type = final_state_dict.get("request_type", "unknown")
        
        if request_type == "design_building":
            # Format design results
            box = final_state_dict.get("box", {})
            is_compliant = final_state_dict.get("compliant", False)
            issues = final_state_dict.get("issues", [])
            
            response_text = "Building Design Results:\n"
            response_text += f"Compliant: {is_compliant}\n"
            
            if issues:
                response_text += f"Issues: {', '.join(issues)}\n"
                
            response_text += "\nDimensions:\n"
            for key, value in box.items():
                if value is not None:
                    if isinstance(value, float):
                        response_text += f"- {key}: {value:.2f}\n"
                    else:
                        response_text += f"- {key}: {value}\n"
            
            return {
                "response": response_text,
                "type": "design",
                "data": {
                    "box": box,
                    "compliant": is_compliant,
                    "issues": issues
                }
            }
            
        elif request_type == "show_guide":
            # Return the design guidelines
            return {
                "response": final_state_dict.get("answer", "No guidelines available."),
                "type": "guide",
                "data": None
            }
            
        elif request_type == "general_question":
            # Return the answer to the question
            answer = final_state_dict.get("answer", "No answer available.")
            search_results = final_state_dict.get("search_results")
            
            return {
                "response": answer,
                "type": "answer",
                "data": {"search_results": search_results} if search_results else None
            }
            
        else:
            # Unknown request type
            return {
                "response": final_state_dict.get("answer", "I don't understand your request."),
                "type": "unknown",
                "data": None
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    """Return API information."""
    return {
        "message": "Architectural Assistant API",
        "endpoints": {
            "/chat": "POST - Send a message to the assistant",
            "/": "GET - Get API information"
        },
        "example": {
            "request": {"message": "Design a building with 1000 square meters and 3 floors"},
            "response": {"response": "Building Design Results...", "type": "design", "data": {"box": {}}}
        }
    }

if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)