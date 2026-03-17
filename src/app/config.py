import os
import sys
from dotenv import load_dotenv

# Resolve paths relative to this file so they work regardless of cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # AgentApp/

# Load .env.local for secret keys only (API keys, tokens)
load_dotenv(os.path.join(_ROOT, ".env.local"))

# ── Non-secret settings come from settings.py ────────────────────────────────
sys.path.insert(0, _ROOT)
import settings as _s

LLM_PROVIDER    = _s.LLM_PROVIDER.lower()
LLM_ENDPOINT    = _s.LLM_ENDPOINT
LLM_MODEL       = _s.LLM_MODEL
LLM_TEMPERATURE = _s.LLM_TEMPERATURE
LLM_TIMEOUT     = _s.LLM_TIMEOUT
GEMINI_MODEL    = _s.GEMINI_MODEL
MCP_GH_ENDPOINT = _s.MCP_GH_ENDPOINT
MCP_TIMEOUT     = _s.MCP_TIMEOUT

# ── Secret keys (from .env.local only) ───────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
