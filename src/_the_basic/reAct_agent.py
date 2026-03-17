from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

import json
import os
import requests

# Load environment variables instead of importing from keys file
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_KEY = os.getenv("CLOUDFLARE_API_KEY")


class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_steps: int


@tool
def calculate_area(length: float, width: float) -> float:
    """
    :param length: length of the architectural space in meters
    :param width: width of the architectural space in meters
    :return: the area of the space in square meters
    """
    return float(length) * float(width)

@tool
def calculate_volume(length: float, width: float, height: float) -> float:
    """
    :param length: length of the architectural space in meters
    :param width: width of the architectural space in meters
    :param height: height of the architectural space in meters
    :return: the volume of the space in cubic meters
    """
    return float(length) * float(width) * float(height)

tools = [TavilySearch(tavily_api_key=TAVILY_API_KEY, max_results=1), calculate_area, calculate_volume]

system_prompt = SystemMessage(
    """
    You are an architectural reasoning agent. Always think step-by-step.

    Use the following format:

    Thought: what you are thinking
    Action: the action to take, e.g. 'search', 'calculate_area', 'calculate_volume'
    Action Input: the input to the action
    Observation: the result of the action

    (Repeat Thought/Action/Observation if needed)

    Final Answer: your answer to the user

    Question: {input}
    """
)

CLOUDFLARE_MODEL = "@cf/meta/llama-4-scout-17b-16e-instruct"  # or another supported model

def call_cloudflare_workers_ai(messages):
    url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/{CLOUDFLARE_MODEL}"
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_KEY}",
        "Content-Type": "application/json"
    }
    # Format messages as needed by the API
    data = {
        "messages": messages
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["result"]["response"]

# Replace the model definition and call_model function:
def call_model(
    state: AgentState,
    config: RunnableConfig,
):
    # Prepare messages for Cloudflare API
    messages = [{"role": "system", "content": system_prompt.content}]
    for msg in state["messages"]:
        messages.append({"role": msg.type, "content": msg.content})
    response = call_cloudflare_workers_ai(messages)
    # Wrap response as needed for your agent
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=response)]}


tools_by_name = {tool.name: tool for tool in tools}

# Define our tool node
def tool_node(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as agent. This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use 'agent'.
    # This means these are the edges taken after the 'agent' node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next
    should_continue,
    {
        "continue": "tools", # If 'tools', then we call the tool node.
        "end": END, # Otherwise we finish.
    },
)

# We now add a normal edge from 'tools' to 'agent'.
# This means that after 'tools' is called, 'agent' node is called next.
workflow.add_edge("tools", "agent")

# Now we can compile and visualize the graph
from IPython.display import Image, display
import os

# Create a directory for saving the graph if it doesn't exist
graph_dir = os.path.join(os.path.dirname(__file__), "graphs")
os.makedirs(graph_dir, exist_ok=True)

# Compile the graph first
app = workflow.compile()

try:
    # Save the graph as a PNG file
    graph_path = os.path.join(graph_dir, "react_agent_graph.png")
    
    # Generate and save the image using the method from design_agent_simple.py
    with open(graph_path, "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
    
    print(f"Graph image saved to {graph_path}")
    
    # Also display the image if in an IPython environment
    try:
        display(Image(filename=graph_path))
    except:
        pass
except Exception as e:
    print(f"Error generating or saving graph: {e}")

if __name__ == "__main__":
    inputs = {"messages": [("user", "What's the area of a room that is 5 meters by 7 meters? Then, if the ceiling is 3 meters high, what's the volume?")]}
    app = workflow.compile()
    for state in app.stream(inputs, stream_mode="values"):
        last_message = state["messages"][-1]
        last_message.pretty_print()