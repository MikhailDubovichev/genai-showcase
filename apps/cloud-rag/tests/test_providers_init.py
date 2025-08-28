import os
import pytest


@pytest.mark.parametrize("provider", ["nebius", "openai"]) 
def test_build_llm_and_embeddings(provider: str, monkeypatch):
    # Ensure required env variables exist before importing CONFIG (it validates at import time)
    if not os.getenv("NEBIUS_API_KEY"):
        monkeypatch.setenv("NEBIUS_API_KEY", "dummy-nebius-key")
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")

    # Arrange: load cloud CONFIG from local app (tests run inside this dir)
    from config import CONFIG  # type: ignore
    from providers.factory import get_chat_llm, get_embeddings  # type: ignore

    # Mutate CONFIG providers
    CONFIG["llm"]["provider"] = provider
    CONFIG["embeddings"]["provider"] = provider

    # Act: construct providers (no network call)
    llm = get_chat_llm(CONFIG)
    emb = get_embeddings(CONFIG)

    # Assert
    assert llm is not None
    assert emb is not None

