# ─────────────────────────────────────────────────────────────────────────────
#  settings.py  —  Runtime configuration (non-secret + secret keys)
#
#  Edit this file to switch LLM providers, models, or toggle plan mode.
#  Secret keys are loaded from .env.local in the same directory.
# ─────────────────────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, ".env.local"))

# ── Secret keys (loaded from .env.local) ─────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ── Cloudflare Workers AI ─────────────────────────────────────────────────────
CF_ACCOUNT_ID = os.getenv("CF_ACCOUNT_ID", "")
CF_API_TOKEN  = os.getenv("CF_API_TOKEN", "")
CF_MODEL      = "@cf/meta/llama-3.1-8b-instruct"

# ── LLM Provider ─────────────────────────────────────────────────────────────
# "local"       → OpenAI-compatible local server (e.g. LM Studio, Ollama)
# "gemini"      → Google Gemini via google-genai SDK
# "cloudflare"  → Cloudflare Workers AI (free tier, no credit card needed)
LLM_PROVIDER = "gemini"

# ── Local LLM (OpenAI-compatible endpoint, e.g. LM Studio) ───────────────────
LLM_ENDPOINT    = "http://localhost:1234/v1/chat/completions"
LLM_MODEL       = None   # None = use server default; or e.g. "llama-3.2-3b-instruct"
LLM_TEMPERATURE = 0.2
LLM_TIMEOUT     = 60     # seconds

# ── Google Gemini ─────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash-lite"

# ── Grasshopper MCP server ────────────────────────────────────────────────────
MCP_GH_ENDPOINT = "http://localhost:5001/mcp/"  # Swiftlet JSON-RPC 2.0 endpoint
MCP_TIMEOUT     = 30     # seconds


