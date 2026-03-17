"""Design Agent — terminal REPL.

Usage (from AgentApp/ directory):
    python run_agent.py

Commands during chat:
    reload   — re-fetch GH tools from the MCP server
    tools    — list currently loaded GH tools
    quit / exit / Ctrl-C  — exit
"""
import os
import sys
import textwrap

# ── Bootstrap path so imports work when run directly ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HR  = "─" * 72
HR2 = "═" * 72


def _check_llm() -> bool:
    """Ping the LLM endpoint (local) or validate the API key (Gemini). Returns True if ready."""
    import requests
    from app.config import LLM_PROVIDER, LLM_ENDPOINT, GOOGLE_API_KEY, GEMINI_MODEL

    if LLM_PROVIDER == "gemini":
        if GOOGLE_API_KEY:
            print(f"  [llm] Gemini  model = {GEMINI_MODEL}  (API key set)")
            return True
        else:
            print("  [llm] Gemini selected but GOOGLE_API_KEY is not set in .env.local!")
            return False

    # Derive the /models probe URL (works for LM Studio, Ollama, OpenAI-compat)
    base = LLM_ENDPOINT.rstrip("/").removesuffix("/chat/completions").removesuffix("/v1")
    probe = f"{base}/v1/models"
    try:
        resp = requests.get(probe, timeout=3)
        if resp.status_code < 500:
            print(f"  [llm] OK  {LLM_ENDPOINT}")
            return True
    except Exception:
        pass
    print(f"  [llm] UNREACHABLE  {LLM_ENDPOINT}")
    print("        Make sure LM Studio (or your local LLM) is running before sending prompts.")
    return False


def _banner():
    print(HR2)
    print("  Design Agent  (terminal mode)")
    print(HR2)
    print("  Commands:  reload · tools · plan on/off · history · quit")
    print(HR)


def _print_tools():
    try:
        from tools import TOOL_CLASSES
        if not TOOL_CLASSES:
            print("  [tools] No GH tools loaded — is the Grasshopper plugin running?")
        else:
            print(f"  [tools] {len(TOOL_CLASSES)} tool(s) loaded:")
            for t in TOOL_CLASSES:
                print(f"    • {t.name}  —  {t.description[:80]}")
    except Exception as exc:
        print(f"  [tools] error: {exc}")


def _reload_tools():
    try:
        from tools import reload_mcp_tools
        tools = reload_mcp_tools()
        print(f"  [reload] {len(tools)} tool(s) loaded.")
    except Exception as exc:
        print(f"  [reload] error: {exc}")


def _run(graph, user_input: str, messages: list, force_plan: bool = False):
    from models.state import BoxState
    state = BoxState(
        request={"user_input": user_input},
        messages=list(messages),
        # Skip classifier — route straight to planner when plan mode is on
        request_type="plan" if force_plan else None,
    )
    print()
    print(f"  ┊ input: {user_input}")
    try:
        result = graph.invoke(state, config={"recursion_limit": 50})
    except Exception as exc:
        print(f"\n  [error] {exc}")
        return None

    answer = result.get("answer") or ""
    request_type  = result.get("request_type", "?")
    tool_results  = result.get("tool_results") or {}
    plan_results  = result.get("plan_results") or {}

    print(f"\n[{request_type}]")
    print(HR)

    if answer:
        answer_str = answer if isinstance(answer, str) else "\n".join(str(a) for a in answer)
        for line in textwrap.wrap(answer_str, width=70) or [answer_str]:
            print(" ", line)
    elif result.get("box"):
        box = result["box"]
        print(f"  Compliant : {result.get('compliant')}")
        issues = result.get("issues") or []
        print(f"  Issues    : {', '.join(issues) if issues else 'none'}")
        print("  Box       :")
        for k, v in box.items():
            print(f"    {k}: {v}")
    else:
        print("  (no answer)")

    if tool_results:
        print(HR)
        print("  Tool results:")
        for name, val in tool_results.items():
            print(f"    [{name}] {str(val)[:200]}")

    if plan_results:
        print(HR)
        print("  Plan step results:")
        for key, val in plan_results.items():
            print(f"    [{key}] {str(val)[:200]}")

    print(HR)
    return (answer if isinstance(answer, str) else "\n".join(str(a) for a in answer)) if answer else None


def main():
    _banner()
    _check_llm()
    _print_tools()
    print()

    from graphs.main_graph import build_main_graph
    graph = build_main_graph()

    # ── save + open graph image on startup ───────────────────────────────────
    try:
        os.makedirs("visualizations", exist_ok=True)
        png_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualizations", "graph.png")
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open(png_path, "wb") as f:
            f.write(png_bytes)
        print(f"  [graph] saved → {png_path}")
    except Exception as exc:
        print(f"  [graph] could not render image: {exc}")
    print()

    conversation_messages: list = []   # persisted across turns in this session

    import settings as _s
    plan_mode: bool = _s.PLAN_MODE       # default from settings.py; toggled via 'plan on/off'

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        elif user_input.lower() == "reload":
            _reload_tools()
            continue
        elif user_input.lower() == "tools":
            _print_tools()
            continue
        elif user_input.lower() in ("history", "mem"):
            if not conversation_messages:
                print("  (no conversation history yet)")
            else:
                for m in conversation_messages:
                    tag = "You" if m["role"] == "user" else "Agent"
                    print(f"  {tag}: {m['content'][:120]}")
            print()
            continue
        elif user_input.lower() in ("plan on", "plan"):
            plan_mode = True
            print("  [plan mode] ON  — prompts will be decomposed into multi-tool sequences.")
            print("                    Type 'plan off' to return to normal routing.")
            print()
            continue
        elif user_input.lower() == "plan off":
            plan_mode = False
            print("  [plan mode] OFF  — back to normal routing.")
            print()
            continue
        elif user_input.lower() == "plan status":
            print(f"  [plan mode] {'ON' if plan_mode else 'OFF'}")
            print()
            continue

        if plan_mode:
            print("  [plan mode ON]")

        answer = _run(graph, user_input, conversation_messages, force_plan=plan_mode)

        # Accumulate conversation memory (keep last 20 turns to avoid unbounded growth)
        conversation_messages.append({"role": "user", "content": user_input})
        if answer:
            conversation_messages.append({"role": "assistant", "content": answer})
        if len(conversation_messages) > 40:  # 20 turns × 2
            conversation_messages = conversation_messages[-40:]


if __name__ == "__main__":
    main()


