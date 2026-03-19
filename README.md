# AIA26 Studio — LangGraph Agent

A LangGraph-based architectural assistant that connects an LLM to **Grasshopper / Rhino** via a local MCP (Model Context Protocol) server.  
The agent classifies user requests, commits to a design scope, and routes them through specialised branches: building design with code-compliance checking, climate optimisation, web-augmented Q&A, and general architectural guidance.

---

## Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.11 |
| Rhino 8 + Grasshopper | Running with the **Swiftlet** plugin active on port `5001` |
| LLM API key | Pick **one** of the three options below |
| Tavily API key | Optional — enables web search in the `general_question` branch ([tavily.com](https://tavily.com)) |

### LLM options (pick one)

| Provider | Cost | Sign-up |
|---|---|---|
| **Google Gemini** (`LLM_PROVIDER=gemini`) | Free tier | [aistudio.google.com](https://aistudio.google.com) — no credit card |
| **Cloudflare Workers AI** (`LLM_PROVIDER=cloudflare`) | Free tier | [dash.cloudflare.com](https://dash.cloudflare.com) — no credit card |
| **Local server** (`LLM_PROVIDER=local`) | Free | LM Studio or Ollama running locally |

> **No Conda?** The steps below show `conda`, but a plain Python `venv` works too — see the venv alternative at the end of step 1.

---

## Setup

**1. Clone and create the environment**

_Option A — conda (recommended if you have Anaconda / Miniconda):_
```bash
git clone <repo-url>
cd AIA26_Studio_LangGraph
conda create -n 311 python=3.11 -y
conda activate 311
```

_Option B — plain Python venv (no conda required):_
```bash
git clone <repo-url>
cd AIA26_Studio_LangGraph
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r src/requirements.txt
```

**3. Get your API key and configure secrets**

Create a file called `.env.local` inside the `src/` folder (this file is gitignored and never committed).

_Option A — Google Gemini:_
1. Go to [aistudio.google.com](https://aistudio.google.com), sign in, and click **Get API key**.

```
GOOGLE_API_KEY=paste_your_key_here
TAVILY_API_KEY=your_tavily_key        # optional
```

_Option B — Cloudflare Workers AI:_
1. Log in to [dash.cloudflare.com](https://dash.cloudflare.com).
2. Go to **Workers & Pages → Overview** and copy your **Account ID** from the right sidebar.
3. Go to **My Profile → API Tokens → Create Token → Use "Read all resources" template**, then restrict it to **Account → Workers AI → Edit**.
4. Copy the generated token.

```
CF_ACCOUNT_ID=your_account_id_here
CF_API_TOKEN=your_api_token_here
TAVILY_API_KEY=your_tavily_key        # optional
```

Then in `src/settings.py` set `LLM_PROVIDER = "cloudflare"`.

_Option C — Local server (LM Studio / Ollama):_

No API key required. Start your local server and set `LLM_PROVIDER = "local"` in `src/settings.py`.

**4. Configure settings** (optional overrides in `src/settings.py`)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `"gemini"` | `"gemini"`, `"cloudflare"`, or `"local"` |
| `GEMINI_MODEL` | `"gemini-2.5-flash-lite"` | Gemini model name |
| `CF_MODEL` | `"@cf/meta/llama-3.1-8b-instruct"` | Cloudflare Workers AI model name |
| `LLM_ENDPOINT` | `http://localhost:1234/v1/...` | Local server URL (only when `LLM_PROVIDER="local"`) |
| `MCP_GH_ENDPOINT` | `http://localhost:5001/mcp/` | Swiftlet JSON-RPC endpoint |
| `PLAN_MODE` | `False` | Force multi-step plan routing globally |

For most students, the defaults are fine — just set your API key and you're ready.

**5. Start Rhino + Grasshopper**

1. Open **Rhino 8**.
2. Make sure the **Swiftlet** plugin is installed and enabled.
3. Open the provided Grasshopper definition: `src/gh_tools/create_building.gh`.
4. Confirm the MCP server is listening on port `5001` (Swiftlet shows a status indicator).

**6. Run**
```bash
# Interactive REPL
cd src
python run_agent.py

# Or open the Jupyter notebook (install jupyterlab first if needed: pip install jupyterlab)
jupyter lab src/notebooks/agent_system_tests.ipynb
```

---

## REPL Commands

Once `run_agent.py` is running, use these special commands at the prompt:

| Command | Effect |
|---|---|
| `quit` / `exit` | Exit the REPL |
| `history` | Print all turns in the current conversation thread |
| `tools` | List all tools loaded from the MCP server |
| `plan on` | Enable plan mode (force multi-step planner routing) |
| `plan off` | Disable plan mode |
| `plan status` | Show current plan mode setting |
| `reload` | Reload the graph and all settings without restarting |

---

## Folder Structure

```
AIA26_Studio_LangGraph/
├── README.md
└── src/
    ├── run_agent.py          # Terminal REPL entry point
    ├── settings.py           # All runtime config + secret key loading
    ├── requirements.txt      # Python dependencies
    │
    ├── config/
    │   ├── design_rules.py   # Building code constraints (area, height, width, ratio…)
    │   └── prompts.py        # Shared prompt templates
    │
    ├── graphs/
    │   └── main_graph.py     # Builds and wires the LangGraph StateGraph
    │
    ├── models/
    │   └── state.py          # AgentState — Pydantic model shared across all nodes
    │
    ├── nodes/
    │   ├── classification_node.py       # LLM-based router → request types
    │   ├── building_design_node.py      # Planner + ReAct loop:
    │   │                               #   building_planner → retrieve_rules →
    │   │                               #   thinking → execute_action → draw_box (MCP) →
    │   │                               #   compliance_check → is_compliant
    │   ├── climate_optimization_node.py # Climate / energy analysis pipeline
    │   ├── information_node.py          # show_guide, handle_unknown
    │   └── search_node.py              # determine_search_need → [web_search] → answer
    │
    ├── tools/
    │   ├── base.py                      # BaseAgentTool — base class for all tools
    │   ├── mcp/
    │   │   ├── loader.py                # Discovers Grasshopper tools from the MCP server
    │   │   └── list_tools.py            # CLI helper to list available MCP tools
    │   └── search/
    │       └── tools.py                 # WebSearchTool wrapping Tavily
    │
    ├── utils/
    │   └── llm_utils.py                 # LLM wrappers (Gemini, Cloudflare, local)
    │
    ├── gh_tools/
    │   └── create_building.gh           # Grasshopper definition for the building tool
    │
    ├── notebooks/
    │   └── agent_system_tests.ipynb     # End-to-end test notebook
    │
    └── visualizations/                  # Graph visualisation outputs
```

---

## Agent Architecture

Every request goes through a two-layer workflow:

### Layer 1 — Classification
`classify_input` uses an LLM prompt to categorise the input into one of these request types:

| `request_type` | Description |
|---|---|
| `design_building` | Size and draw a code-compliant building |
| `climate_optimization` | Analyse or optimise a building for climate / energy |
| `use_tool` | Run a single named Grasshopper tool |
| `show_guide` | Return project design rules and constraints |
| `general_question` | Architecture Q&A, optionally with Tavily web search |
| `unknown` | Politely decline off-topic requests |

### Layer 2 — Plan & Execute (building design branch)
Before the ReAct loop runs, a **`building_planner`** node commits to a scope:

- **`"compliance only"`** — size the building against code rules, draw it in Grasshopper, stop.
- **`"compliance + climate optimization"`** — same as above, then continue to the climate pipeline.

The planner stores its scope decision and step list in `state.context["build_plan"]` so the rest of the graph can follow the committed plan.

### Full routing map

```
classify_input
  │
  ├── design_building ──────► building_planner
  │                                 │
  │                       ┌─────────┴──────────────────────┐
  │                       │                                │
  │               [compliance only]        [compliance + climate]
  │                       │                                │
  │               retrieve_rules ◄─────────────────────────┘
  │                       │
  │               thinking → execute_action → draw_box (MCP)
  │                       │
  │               compliance_check → is_compliant
  │                       │
  │              ┌─────────┴────────────────┐
  │           [done]                [climate_optimization]
  │                                         │
  │                               climate_investigate
  │                                         │
  │                               climate_optimize → done
  │
  ├── use_tool ────────────► execute_gh_tool → done
  ├── show_guide ──────────► show_guide → done
  ├── general_question ────► determine_search_need → [web_search →] answer → done
  └── unknown ─────────────► handle_unknown → done
```

---

## Key Files

| File | Purpose |
|---|---|
| `src/settings.py` | Single source of truth for provider, model, endpoints, and API keys |
| `src/models/state.py` | `AgentState` — the shared Pydantic state object passed through every node |
| `src/graphs/main_graph.py` | Assembles all nodes and edges; call `build_main_graph(checkpointer=…)` |
| `src/nodes/classification_node.py` | Prompt-based LLM router classifying input into request types |
| `src/nodes/building_design_node.py` | `building_planner` scope decision + full ReAct compliance loop |
| `src/config/design_rules.py` | `DESIGN_GUIDE` list — max area, height, width, floor count, ratios, exits |
| `src/tools/mcp/loader.py` | Discovers Grasshopper tools from the running MCP server |
| `src/utils/llm_utils.py` | LLM adapters for Gemini, Cloudflare Workers AI, and local servers |
| `src/run_agent.py` | Interactive terminal REPL |
| `src/notebooks/agent_system_tests.ipynb` | End-to-end test notebook (all routing branches, classifier validation, multi-turn memory demo) |


---

## Prerequisites

| Requirement | Details |
|---|---|
| Python | 3.11 |
| Rhino 8 + Grasshopper | Running with the **Swiftlet** plugin active on port `5001` |
| Google Gemini API key | Free tier at [aistudio.google.com](https://aistudio.google.com) — no credit card needed |
| Tavily API key | Optional — enables web search in the `general_question` branch ([tavily.com](https://tavily.com)) |

> **No Conda?** The steps below show `conda`, but a plain Python `venv` works too — see the venv alternative at the end of step 1.

---

## Setup

**1. Clone and create the environment**

_Option A — conda (recommended if you have Anaconda / Miniconda):_
```bash
git clone <repo-url>
cd AIA26_Studio_LangGraph
conda create -n 311 python=3.11 -y
conda activate 311
```

_Option B — plain Python venv (no conda required):_
```bash
git clone <repo-url>
cd AIA26_Studio_LangGraph
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

**2. Install dependencies**
```bash
pip install -r src/requirements.txt
```

**3. Get your API key and configure secrets**

1. Go to [aistudio.google.com](https://aistudio.google.com), sign in with a Google account, and click **Get API key**.
2. Create a file called `.env.local` inside the `src/` folder (this file is gitignored and never committed):

```
GOOGLE_API_KEY=paste_your_key_here
TAVILY_API_KEY=your_tavily_key        # optional — leave blank or omit if not needed
```

**4. Configure settings** (optional overrides in `src/settings.py`)

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `"gemini"` | `"gemini"` or `"local"` (LM Studio / Ollama) |
| `GEMINI_MODEL` | `"gemini-2.5-flash-lite"` | Gemini model name |
| `LLM_ENDPOINT` | `http://localhost:1234/v1/...` | Local server URL (only used when `LLM_PROVIDER="local"`) |
| `MCP_GH_ENDPOINT` | `http://localhost:5001/mcp/` | Swiftlet JSON-RPC endpoint |
| `PLAN_MODE` | `False` | Force multi-step plan routing globally |

For most students, the defaults are fine — just set your `GOOGLE_API_KEY` and you're ready.

**5. Start Rhino + Grasshopper**

1. Open **Rhino 8**.
2. Make sure the **Swiftlet** plugin is installed and enabled.
3. Open the provided Grasshopper definition: `src/gh_tools/create_building.gh`.
4. Confirm the MCP server is listening on port `5001` (Swiftlet shows a status indicator).

**6. Run**
```bash
# Interactive REPL
cd src
python run_agent.py

# Or open the Jupyter notebook (install jupyterlab first if needed: pip install jupyterlab)
jupyter lab src/notebooks/agent_system_tests.ipynb
```

---

## Folder Structure

```
AIA26_Studio_LangGraph/
├── README.md
└── src/
    ├── run_agent.py          # Terminal REPL entry point
    ├── settings.py           # All runtime config + secret key loading
    ├── requirements.txt      # Python dependencies
    │
    ├── config/
    │   ├── design_rules.py   # Building code constraints (area, height, width, ratio…)
    │   └── prompts.py        # Shared prompt templates
    │
    ├── graphs/
    │   └── main_graph.py     # Builds and wires the LangGraph StateGraph
    │
    ├── models/
    │   └── state.py          # BoxState — Pydantic model shared across all nodes
    │
    ├── nodes/
    │   ├── classification_node.py       # LLM-based router → 6 request types
    │   ├── building_design_node.py      # ReAct loop: retrieve_rules → thinking →
    │   │                               #   execute_action → draw_box (MCP) →
    │   │                               #   compliance_check → is_compliant
    │   ├── information_node.py          # show_guide, handle_unknown
    │   ├── planning_node.py             # planner → execute_plan_step (loop) → plan_summary
    │   └── search_node.py              # determine_search_need → [web_search] → answer
    │
    ├── tools/
    │   ├── base.py                      # BaseAgentTool — base class for all tools
    │   ├── mcp/
    │   │   ├── loader.py                # Dynamically loads Grasshopper tools from the
    │   │   │                            #   MCP server at import time (JSON-RPC 2.0)
    │   │   └── list_tools.py            # CLI helper to list available MCP tools
    │   └── search/
    │       └── tools.py                 # WebSearchTool wrapping Tavily
    │
    ├── utils/
    │   └── llm_utils.py                 # LLM wrapper (Gemini + local OpenAI-compatible)
    │
    ├── gh_tools/
    │   └── create_building.gh           # Grasshopper definition for the building tool
    │
    ├── notebooks/
    │   └── agent_system_tests.ipynb     # End-to-end test notebook (all 6 routing branches)
    │
    └── visualizations/                  # Graph visualisation outputs
```

---

## Agent Graph

The graph entry point is always `classify_input`, which routes to one of six branches:

| `request_type` | Route | Description |
|---|---|---|
| `design_building` | `retrieve_rules → thinking → execute_action → draw_box → compliance_check → is_compliant` | Sizes a building against code rules using a ReAct loop; calls Grasshopper on each iteration |
| `plan` | `planner → execute_plan_step (loop) → plan_summary` | Decomposes a multi-step prompt into ordered tool calls |
| `use_tool` | `execute_gh_tool` | Runs a single named Grasshopper tool |
| `show_guide` | `show_guide` | Returns the project design rules and constraints |
| `general_question` | `determine_search_need → [perform_web_search →] answer` | Answers architecture Q&A; optionally performs a Tavily web search for current information |
| `unknown` | `handle_unknown` | Politely declines off-topic requests |

---

## Key Files

| File | Purpose |
|---|---|
| `src/settings.py` | Single source of truth for provider, model, endpoints, and API keys |
| `src/models/state.py` | `BoxState` — the shared Pydantic state object passed through every node |
| `src/graphs/main_graph.py` | Assembles all nodes and edges; call `build_main_graph(checkpointer=…)` |
| `src/nodes/classification_node.py` | Prompt-based LLM router classifying input into one of 6 types |
| `src/config/design_rules.py` | `DESIGN_GUIDE` list — max area, height, width, floor count, ratios, exits |
| `src/tools/mcp/loader.py` | Discovers Grasshopper tools from the running MCP server; exposes them as LangChain tools |
| `src/utils/llm_utils.py` | `fast_llm()` helper + `ChatLocalLLM` / Gemini adapter |
| `src/run_agent.py` | Interactive terminal REPL with `reload` / `tools` commands |
| `src/notebooks/agent_system_tests.ipynb` | Notebook with direct MCP tests, per-branch agent runs, classifier validation, graph visualisation, and multi-turn memory demo |


