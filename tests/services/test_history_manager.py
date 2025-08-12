"""
Tests for `services/history_manager.py` using pytest.

Focus:
- ID generation and user hashing
- Path resolution for active conversations (global and user-specific)
- Load behavior with missing/invalid files
- Save behavior and persistence
- Archive behavior for global and user-specific conversations

We isolate filesystem effects by redirecting `CONVERSATIONS_DIR` to a `tmp_path` in a fixture.
"""

import json
import os
import re
from pathlib import Path

import pytest

import services.history_manager as hm


@pytest.fixture(autouse=True)
def tmp_conversations_dir(tmp_path, monkeypatch):
    """
    Redirect the history manager to use a temporary conversations directory for each test.

    This prevents writes to the real project `user_data` folder and ensures tests are isolated.
    """
    conv_dir = tmp_path / "conversations"
    conv_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(hm, "CONVERSATIONS_DIR", str(conv_dir), raising=True)
    # Ensure ACTIVE_CONVERSATION_FILENAME is stable
    monkeypatch.setattr(hm, "ACTIVE_CONVERSATION_FILENAME", "active_conversation.json", raising=True)
    yield conv_dir


def test_generate_interaction_id_uuid():
    iid = hm.generate_interaction_id()
    assert isinstance(iid, str)
    # UUID v4 pattern (relaxed)
    assert re.match(r"^[0-9a-fA-F\-]{36}$", iid)


def test_get_user_hash_is_lowercase_and_deterministic():
    h1 = hm.get_user_hash("User@Example.com")
    h2 = hm.get_user_hash("user@example.com")
    assert h1 == h2
    assert len(h1) == 16
    # hex chars only
    assert re.match(r"^[0-9a-f]+$", h1)


def test_get_active_conversation_path_global(tmp_conversations_dir):
    path = hm.get_active_conversation_path(None)
    assert Path(path).parent == Path(tmp_conversations_dir)
    assert Path(path).name == "active_conversation.json"


def test_get_active_conversation_path_user_specific(tmp_conversations_dir):
    email = "user@example.com"
    path = hm.get_active_conversation_path(email)
    # filename starts with hash and ends with _active_conversation.json
    assert Path(path).parent == Path(tmp_conversations_dir)
    assert Path(path).name.endswith("_active_conversation.json")
    # ensure hash prefix matches function
    expected_prefix = hm.get_user_hash(email)
    assert Path(path).name.startswith(expected_prefix)


def test_load_conversation_history_missing_returns_empty():
    data = hm.load_conversation_history(user_email=None)
    assert data == []


def test_load_conversation_history_invalid_json_returns_empty(tmp_conversations_dir):
    bad_file = Path(tmp_conversations_dir) / "active_conversation.json"
    bad_file.write_text("not json", encoding="utf-8")
    data = hm.load_conversation_history(user_email=None)
    assert data == []


def test_save_and_load_message_global(tmp_conversations_dir):
    iid = hm.generate_interaction_id()
    saved = hm.save_message(
        interaction_id=iid,
        role="user",
        content="hello",
        user_email=None,
    )
    assert saved["interaction_id"] == iid
    path = hm.get_active_conversation_path(None)
    assert Path(path).exists()

    history = hm.load_conversation_history(user_email=None)
    assert isinstance(history, list) and len(history) == 1
    assert history[0]["content"] == "hello"
    assert history[0]["role"] == "user"


def test_save_and_load_message_user_specific(tmp_conversations_dir):
    email = "user@example.com"
    iid = hm.generate_interaction_id()
    hm.save_message(
        interaction_id=iid,
        role="assistant",
        content="hi",
        user_email=email,
    )
    path = hm.get_active_conversation_path(email)
    assert Path(path).exists()
    # ensure file name contains user hash
    assert Path(path).name.startswith(hm.get_user_hash(email))

    history = hm.load_conversation_history(user_email=email)
    assert len(history) == 1 and history[0]["content"] == "hi"


def test_archive_active_conversation_no_file_returns_true_message(tmp_conversations_dir):
    ok, msg = hm.archive_active_conversation(user_email=None)
    assert ok is True
    assert isinstance(msg, str)


def test_archive_active_conversation_global_moves_file(tmp_conversations_dir):
    iid = hm.generate_interaction_id()
    hm.save_message(iid, "user", "hello", user_email=None)
    active_path = Path(hm.get_active_conversation_path(None))
    assert active_path.exists()

    ok, msg = hm.archive_active_conversation(user_email=None)
    assert ok is True
    assert "conversation_" in msg or "archived" in msg
    assert not active_path.exists()

    # Ensure an archive file was created
    archived_files = list(Path(tmp_conversations_dir).glob("conversation_*.json"))
    assert len(archived_files) == 1


def test_archive_active_conversation_user_specific_moves_file(tmp_conversations_dir):
    email = "user@example.com"
    iid = hm.generate_interaction_id()
    hm.save_message(iid, "assistant", "hi", user_email=email)
    active_path = Path(hm.get_active_conversation_path(email))
    assert active_path.exists()

    ok, msg = hm.archive_active_conversation(user_email=email)
    assert ok is True
    assert not active_path.exists()

    # Ensure an archive file with user hash prefix was created
    user_hash = hm.get_user_hash(email)
    archived_files = list(Path(tmp_conversations_dir).glob(f"{user_hash}_conversation_*.json"))
    assert len(archived_files) == 1 