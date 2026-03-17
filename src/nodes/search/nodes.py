from models.state import BoxState
from utils.llm_utils import llm, fast_llm
from tools.search import search_web
import json
import textwrap

def _think(label: str, text: str):
    prefix = f"  ┊ {label}: "
    body = str(text).strip().replace("\n", " ")
    for i, line in enumerate(textwrap.wrap(body, width=68)):
        print((prefix if i == 0 else " " * len(prefix)) + line)

def determine_search_need_fn(state: BoxState) -> BoxState:
    """Determine if a web search is needed to answer the general question."""
    user_input = state.request.get("user_input", "")
    
    prompt = f"""
    Analyze this architecture-related question: "{user_input}"
    
    Do you need to search for current information on the web to answer it accurately?
    For example, questions about current building codes, new construction techniques, 
    or recent architectural trends would require a web search.
    
    Questions about general architectural principles, basic design concepts, or 
    standard practices can be answered without a search.
    
    Respond with only "Yes" or "No".
    """
    
    response = fast_llm(prompt)    # yes/no — short timeout is fine
    search_needed = "yes" in str(response).lower()

    state.needs_search = search_needed
    _think("search needed", str(search_needed))

    if search_needed:
        # Create a search query based on the user input
        prompt = f"""
        Transform this user question into a concise web search query related to architecture:
        "{user_input}"

        Return only the search query with no additional text.
        """

        search_query = str(llm(prompt)).strip()
        state.search_query = search_query
        _think("search query", search_query)
    
    state.history.append({
        "node": "determine_search_need",
        "needs_search": state.needs_search,
        "search_query": state.search_query if state.needs_search else None
    })
    
    return state

def perform_web_search_fn(state: BoxState) -> BoxState:
    """Perform a web search using Tavily search."""
    query = state.search_query
    
    try:
        # Use the search_web tool
        results = search_web.invoke(query)
        
        # Check if results contains a list of search items
        if "results" in results and isinstance(results["results"], list):
            state.search_results = results["results"]
        else:
            state.search_results = []
        
        state.history.append({
            "node": "perform_web_search",
            "query": query,
            "results_count": len(state.search_results)
        })
    except Exception as e:
        state.search_results = []
        state.history.append({
            "node": "perform_web_search",
            "query": query,
            "error": str(e)
        })
    
    return state

def answer_with_search_fn(state: BoxState) -> BoxState:
    """Answer general questions using web search results."""
    user_input = state.request.get("user_input", "")
    search_results = state.search_results or []
    
    # Format search results for the prompt
    formatted_results = ""
    for i, result in enumerate(search_results):
        formatted_results += f"Source {i+1}: {result.get('title', 'No title')}\n"
        formatted_results += f"URL: {result.get('url', 'No URL')}\n"
        formatted_results += f"Content: {result.get('content', 'No content')}\n\n"
    
    prompt = f"""
    You are an architectural assistant that helps with building design questions.
    
    User question: "{user_input}"
    
    Here are relevant search results to help answer this question:
    
    {formatted_results}
    
    Using the information from these search results, provide a comprehensive, accurate answer.
    Include relevant facts, figures, and recommendations.
    Cite your sources by referring to them as [Source 1], [Source 2], etc.
    If the search results don't fully address the question, clearly state what information is missing.
    """
    
    answer = str(llm(prompt))
    state.answer = answer
    state.done = True
    
    state.history.append({
        "node": "answer_with_search",
        "answer": answer
    })
    
    return state

def answer_without_search_fn(state: BoxState) -> BoxState:
    """Answer general questions about architecture without web search."""
    user_input = state.request.get("user_input", "")
    
    prompt = f"""
    You are an architectural assistant that helps with building design questions.
    
    User question: "{user_input}"
    
    Provide a helpful, informative response using your knowledge of architecture and building design.
    Focus on timeless architectural principles, standard practices, and established design concepts.
    Include specific details, examples, and recommendations where appropriate.
    If the question requires current data or recent trends that you don't have access to, 
    acknowledge this limitation in your response.
    """
    
    answer = str(llm(prompt))

    state.answer = answer
    state.done = True

    state.history.append({
        "node": "answer_without_search",
        "answer": answer
    })

    return state
