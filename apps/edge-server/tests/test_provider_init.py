import os
import pytest


def _have_key(provider: str) -> bool:
    if provider == "nebius":
        return bool(os.getenv("LLM_API_KEY") or os.getenv("NEBIUS_API_KEY"))
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    return False


@pytest.mark.parametrize("provider", ["nebius", "openai"]) 
def test_get_client_builds(provider: str, monkeypatch):
    from config import CONFIG  # type: ignore
    from llm_cloud.provider import get_client  # type: ignore

    if not _have_key(provider):
        pytest.skip(f"Missing API key for {provider}")

    # Switch provider in-memory
    CONFIG["llm"]["provider"] = provider

    client = get_client()
    assert client is not None

