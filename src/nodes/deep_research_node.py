"""
deep_research_node.py  —  Deep Research Agent for general_question branch

Flow
────
  deep_decompose   →  deep_search   →  deep_reflect  ─→ (gaps?)  deep_followup  →  deep_synthesize
                                                       └→ (done)  deep_synthesize

Stages
──────
1. deep_decompose   : Break the user question into 2-4 targeted sub-questions.
2. deep_search      : Run a Tavily search for each sub-question, accumulate results.
3. deep_reflect     : Evaluate coverage gaps; if gaps exist and iterations < MAX_DEPTH,
                      generate follow-up queries (max 2 extra).
4. deep_followup    : Search follow-up queries and append to accumulated results.
5. deep_synthesize  : Merge everything into a final, cited answer.

Routing
───────
  deep_reflect_router(state) → "follow_up" | "synthesize"
"""

import json
import textwrap

from models.state import AgentState
from utils.llm_utils import llm, fast_llm
from tools.search.tools import search_web

# ── Helpers ───────────────────────────────────────────────────────────────────

MAX_DEPTH = 2   # maximum search-reflect iterations


def _think(label: str, text: str) -> None:
    prefix = f"  ┊ {label}: "
    body = str(text).strip().replace("\n", " ")
    for i, line in enumerate(textwrap.wrap(body, width=68)):
        print((prefix if i == 0 else " " * len(prefix)) + line)


def _search(query: str) -> list:
    """Run a single Tavily search and return a flat list of result dicts."""
    try:
        raw = search_web.invoke(query)
        if isinstance(raw, dict) and "results" in raw:
            return raw["results"] or []
        if isinstance(raw, list):
            return raw
    except Exception as exc:
        _think("search error", str(exc))
    return []


# ── Node 1: Decompose ─────────────────────────────────────────────────────────

def deep_decompose_fn(state: AgentState) -> AgentState:
    """Break the user question into 2–4 focused sub-questions."""
    user_input = state.request.get("user_input", "")

    prompt = f"""You are a research planning assistant for an architectural AI.

User question: "{user_input}"

Break this question into 2 to 4 specific sub-questions that, together, fully cover the topic.
Each sub-question should be a standalone web-search query.

Return ONLY a JSON array of strings. Example:
["sub-question 1", "sub-question 2", "sub-question 3"]"""

    response = str(llm(prompt)).strip()

    # Parse JSON; fall back to the original question if parsing fails
    try:
        start = response.index("[")
        end   = response.rindex("]") + 1
        sub_questions: list = json.loads(response[start:end])
        if not isinstance(sub_questions, list) or not sub_questions:
            raise ValueError("empty")
    except Exception:
        sub_questions = [user_input]

    state.context["sub_questions"]        = sub_questions
    state.context["all_search_results"]   = []
    state.context["deep_research_depth"]  = 0

    for q in sub_questions:
        _think("sub-question", q)

    state.history.append({
        "node": "deep_decompose",
        "sub_questions": sub_questions,
    })
    return state


# ── Node 2: Search ────────────────────────────────────────────────────────────

def deep_search_fn(state: AgentState) -> AgentState:
    """Search every sub-question and accumulate results."""
    sub_questions: list  = state.context.get("sub_questions", [])
    all_results:   list  = state.context.get("all_search_results", [])

    for q in sub_questions:
        _think("searching", q)
        hits = _search(q)
        for r in hits:
            r["_query"] = q          # tag each result with the query that found it
        all_results.extend(hits)
        _think("hits", str(len(hits)))

    state.context["all_search_results"]  = all_results
    state.context["deep_research_depth"] = state.context.get("deep_research_depth", 0) + 1

    state.history.append({
        "node": "deep_search",
        "total_results": len(all_results),
        "depth": state.context["deep_research_depth"],
    })
    return state


# ── Node 3: Reflect ───────────────────────────────────────────────────────────

def deep_reflect_fn(state: AgentState) -> AgentState:
    """Assess coverage gaps; produce follow-up queries if gaps remain."""
    user_input   = state.request.get("user_input", "")
    all_results  = state.context.get("all_search_results", [])
    depth        = state.context.get("deep_research_depth", 1)

    # Summarise the search results for the LLM
    snippet = "\n".join(
        f"[{i+1}] {r.get('title','')}: {str(r.get('content',''))[:200]}"
        for i, r in enumerate(all_results[:12])
    )

    prompt = f"""You are a research quality reviewer for an architectural AI.

Original question: "{user_input}"

Collected search results so far:
{snippet}

Are there significant knowledge gaps that would prevent a complete, accurate answer?
If YES, list 1–2 concise follow-up search queries (as a JSON array).
If NO, return an empty array [].

Respond with ONLY a JSON array. Examples:
  ["follow-up query 1", "follow-up query 2"]
  []"""

    response = str(fast_llm(prompt)).strip()

    follow_up: list = []
    try:
        start = response.index("[")
        end   = response.rindex("]") + 1
        follow_up = json.loads(response[start:end])
        if not isinstance(follow_up, list):
            follow_up = []
    except Exception:
        follow_up = []

    has_gaps = bool(follow_up) and depth < MAX_DEPTH

    state.context["follow_up_queries"] = follow_up if has_gaps else []
    state.context["has_gaps"]          = has_gaps

    _think("gaps", str(has_gaps))
    for q in follow_up:
        _think("follow-up", q)

    state.history.append({
        "node": "deep_reflect",
        "has_gaps": has_gaps,
        "follow_up_queries": follow_up,
        "depth": depth,
    })
    return state


def deep_reflect_router(state: AgentState) -> str:
    """Route: 'follow_up' when gaps remain, else 'synthesize'."""
    return "follow_up" if state.context.get("has_gaps") else "synthesize"


# ── Node 4: Follow-up search ─────────────────────────────────────────────────

def deep_followup_fn(state: AgentState) -> AgentState:
    """Search follow-up queries identified during reflection."""
    follow_up_queries = state.context.get("follow_up_queries", [])
    all_results       = state.context.get("all_search_results", [])

    for q in follow_up_queries:
        _think("follow-up search", q)
        hits = _search(q)
        for r in hits:
            r["_query"] = q
        all_results.extend(hits)
        _think("hits", str(len(hits)))

    state.context["all_search_results"]  = all_results
    state.context["deep_research_depth"] = state.context.get("deep_research_depth", 1) + 1

    state.history.append({
        "node": "deep_followup",
        "total_results": len(all_results),
        "depth": state.context["deep_research_depth"],
    })
    return state


# ── Node 5: Synthesize ────────────────────────────────────────────────────────

def deep_synthesize_fn(state: AgentState) -> AgentState:
    """Merge all search results into a final comprehensive answer."""
    user_input  = state.request.get("user_input", "")
    all_results = state.context.get("all_search_results", [])

    formatted = ""
    for i, r in enumerate(all_results):
        formatted += (
            f"[Source {i+1}] {r.get('title', 'No title')}\n"
            f"URL: {r.get('url', '')}\n"
            f"Content: {str(r.get('content', ''))[:400]}\n\n"
        )

    prompt = f"""You are an expert architectural AI assistant.

User question: "{user_input}"

You have gathered the following research results across multiple search iterations:

{formatted}

Write a comprehensive, well-structured answer to the user's question.
- Organise with clear headings if the answer is long.
- Cite sources inline as [Source N].
- If a source is directly relevant, quote or paraphrase it briefly.
- If some aspects remain uncertain, state this honestly.
- Tailor the answer for an architect or building designer."""

    answer = str(llm(prompt))
    state.answer = answer
    state.done   = True

    _think("answer length", f"{len(answer)} chars")

    state.history.append({
        "node": "deep_synthesize",
        "total_sources_used": len(all_results),
    })
    return state
