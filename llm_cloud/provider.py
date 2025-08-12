"""
provider.py – External LLM client. Build and return a configured Nebius/OpenAI client
----------------------------------------------------------------
In the overall data-flow this file sits at the infrastructure layer.
It is the single place where we talk to the external LLM platform (Nebius' OpenAI-compatible endpoint).

Why a *provider* module?
• Keeps third-party SDK initialisation separate from business logic.
• Offers a tiny, easily mockable `get_client()` function instead of a
  global singleton.  Tests can monkey-patch this function or inject a fake
  client without importing heavy objects.
• Other internal modules (`chat.py`) simply ask for a client; they do
  not need to know about base URLs or API keys.
"""

from openai import OpenAI
from config import CONFIG, ENV

# NOTE: We deliberately wrap client creation in a **function** instead of a
# module-level global (creation of global instance of the client, CLIENT = OpenAI(...) when module is imported).
# That avoids import-time side effects and lets the caller decide when the client should be built.
# This approach allows to avoid potential side effects, so now:
# - client is not created when module is imported, but only when caller asks for it (if the client is created but isn't used, it consumes resources).
# - for testing, we can easily inject a fake client.


def get_client() -> OpenAI: 
    """
    Build and return a configured OpenAI client targeting the Nebius endpoint.
    
    The provider encapsulates all environment and configuration lookups (base URL, API key, and timeouts)
    so that callers do not need to deal with SDK initialization details. Returning a fresh client instead of
    a global singleton makes tests simpler (the function can be monkey‑patched) and avoids import‑time side
    effects in production.
    
    Returns:
        OpenAI: A ready‑to‑use client configured using `CONFIG` and `ENV` values.
    """
    return OpenAI(
        base_url=CONFIG["llm"]["base_url"],
        api_key=ENV["LLM_API_KEY"],
        timeout=CONFIG["llm"].get("timeout", 30),  # seconds – explicit is better than implicit
    ) 