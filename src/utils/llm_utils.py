"""
LLM wrapper — supports:
  • Local OpenAI-compatible server (LM Studio, Ollama, …)  → LLM_PROVIDER=local
  • Google Gemini (AI Studio)                              → LLM_PROVIDER=gemini

Switch providers by setting LLM_PROVIDER in .env.local.
"""
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import requests
from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import ToolCall

try:
    from app.config import (
        LLM_PROVIDER, LLM_ENDPOINT, LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT,
        GOOGLE_API_KEY, GEMINI_MODEL,
    )
except ImportError:
    from dotenv import load_dotenv
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(_HERE)  # AgentApp/
    load_dotenv(os.path.join(_ROOT, ".env"))
    load_dotenv(os.path.join(_ROOT, ".env.local"), override=True)
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()
    LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:1234/v1/chat/completions")
    LLM_MODEL = os.getenv("LLM_MODEL", None)
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")


class ChatLocalLLM(BaseChatModel):
    """LangChain-compatible chat model for any OpenAI-compatible local endpoint."""

    endpoint: str = LLM_ENDPOINT
    temperature: float = LLM_TEMPERATURE
    model: Optional[str] = LLM_MODEL
    timeout: int = LLM_TIMEOUT
    tools: List[Dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "custom-local-chat-llm"

    def bind_tools(self, tools: Sequence[BaseTool], **kwargs: Any) -> "ChatLocalLLM":
        """Return a copy of this model with tools bound for function calling."""
        openai_tools = [convert_to_openai_tool(tool) for tool in tools]
        return self.__class__(
            endpoint=self.endpoint,
            temperature=self.temperature,
            model=self.model,
            timeout=self.timeout,
            tools=openai_tools,
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        def _to_openai(m: BaseMessage) -> Dict[str, Any]:
            if isinstance(m, HumanMessage):
                return {"role": "user", "content": m.content}
            elif isinstance(m, SystemMessage):
                return {"role": "system", "content": m.content}
            elif isinstance(m, AIMessage):
                msg: Dict[str, Any] = {"role": "assistant", "content": m.content or ""}
                if hasattr(m, "tool_calls") and m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.get("id", tc["name"]),
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("args", {})),
                            },
                        }
                        for tc in m.tool_calls
                    ]
                return msg
            elif isinstance(m, ToolMessage):
                return {
                    "role": "tool",
                    "content": str(m.content),
                    "tool_call_id": getattr(m, "tool_call_id", "unknown"),
                }
            else:
                return {"role": "user", "content": str(m.content)}

        try:
            payload: Dict[str, Any] = {
                "messages": [_to_openai(m) for m in messages],
                "temperature": self.temperature,
            }
            if self.model:
                payload["model"] = self.model
            if stop:
                payload["stop"] = stop
            if self.tools:
                payload["tools"] = self.tools

            resp = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            resp.raise_for_status()

            data = resp.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content: str = message.get("content", "") or ""
            raw_tool_calls = message.get("tool_calls", [])

            tool_calls: List[ToolCall] = []
            for tc in raw_tool_calls:
                func_args = tc["function"]["arguments"]
                if isinstance(func_args, str):
                    try:
                        func_args = json.loads(func_args)
                    except json.JSONDecodeError:
                        func_args = {}
                tool_calls.append(
                    ToolCall(
                        name=tc["function"]["name"],
                        args=func_args,
                        id=tc.get("id", tc["function"]["name"]),
                    )
                )

            ai_msg = (
                AIMessage(content=content, tool_calls=tool_calls)
                if tool_calls
                else AIMessage(content=content)
            )
            return ChatResult(generations=[ChatGeneration(message=ai_msg)])

        except Exception as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc


# ── Backwards-compatible simple call helper ──────────────────────────────────
# Existing node code calls  llm(prompt)  or  llm(prompt, max_tokens=N).
# This shim keeps that interface working.

class _SimpleLLMShim:
    """Thin wrapper: exposes __call__(prompt) -> str and a .chat property."""

    def __init__(self, timeout: Optional[int] = None) -> None:
        self._chat = ChatLocalLLM() if timeout is None else ChatLocalLLM(timeout=timeout)

    def __call__(self, prompt: str, **_: Any) -> str:
        result = self._chat._generate([HumanMessage(content=prompt)])
        return result.generations[0].message.content

    @property
    def chat(self) -> ChatLocalLLM:
        return self._chat


# ── Gemini provider ───────────────────────────────────────────────────────────

def _build_gemini_chat(timeout: Optional[int] = None) -> BaseChatModel:
    """Return a LangChain ChatGoogleGenerativeAI instance for Gemini 2.5 Flash Lite."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ImportError(
            "langchain-google-genai is required for Gemini support. "
            "Run: pip install langchain-google-genai"
        ) from exc

    kwargs: Dict[str, Any] = {
        "model": GEMINI_MODEL,
        "google_api_key": GOOGLE_API_KEY,
        "temperature": LLM_TEMPERATURE,
    }
    if timeout is not None:
        kwargs["request_timeout"] = timeout
    return ChatGoogleGenerativeAI(**kwargs)


class _GeminiShim:
    """Thin wrapper matching _SimpleLLMShim's interface, backed by Gemini."""

    def __init__(self, timeout: Optional[int] = None) -> None:
        self._chat = _build_gemini_chat(timeout)

    def __call__(self, prompt: str, **_: Any) -> str:
        result = self._chat.invoke([HumanMessage(content=prompt)])
        return result.content

    @property
    def chat(self) -> BaseChatModel:
        return self._chat


# ── Active provider (swap by changing LLM_PROVIDER in .env.local) ────────────

def _make_shim(timeout: Optional[int] = None):
    if LLM_PROVIDER == "gemini":
        return _GeminiShim(timeout)
    return _SimpleLLMShim(timeout)


def _make_chat_llm() -> BaseChatModel:
    if LLM_PROVIDER == "gemini":
        return _build_gemini_chat()
    return ChatLocalLLM()


# Re-export with the active provider so all nodes pick up the right backend.
llm      = _make_shim()
fast_llm = _make_shim(timeout=15)
chat_llm = _make_chat_llm()


def reason_about_image(
    base64_png: str,
    question: str = "",
    view_name: str = "",
) -> str:
    """
    Send a base64-encoded PNG from a Rhino viewport capture to the VLM and
    return its text response.

    Parameters
    ----------
    base64_png : str
        Base64-encoded PNG string (from capture_viewport's image_base64 field).
    question : str
        What to ask about the image. Defaults to a general scene description.
    view_name : str
        Optional viewport name to include in the prompt context.
    """
    label = f" ({view_name})" if view_name else ""
    prompt_text = question or (
        f"This is a screenshot of the Rhino 3D viewport{label}. "
        "Describe what geometry you see: types, approximate sizes, positions, "
        "and any observations about the design. Be concise."
    )
    msg = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_png}"},
        },
        {"type": "text", "text": prompt_text},
    ])
    try:
        result = chat_llm._generate([msg])
        return result.generations[0].message.content
    except Exception as exc:
        return f"(VLM reasoning unavailable: {exc})"

print(f"  [llm] provider = {LLM_PROVIDER}"
      + (f"  model = {GEMINI_MODEL}" if LLM_PROVIDER == "gemini"
         else f"  endpoint = {LLM_ENDPOINT}"))
