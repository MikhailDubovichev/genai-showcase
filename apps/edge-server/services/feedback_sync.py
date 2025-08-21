"""
Feedback sync service for the edge server (self-contained, stdlib-only).

This module implements a minimal daily feedback synchronization routine designed
to mirror locally captured user feedback to the Cloud service. It reads feedback
items from well-known local storage locations under the edge app's user data
directory, filters only the items that are newer than the last successful sync
checkpoint, and posts them to the Cloud via a simple HTTP (Hypertext Transfer
Protocol) POST request with a compact JSON (JavaScript Object Notation) body.
The configuration values are read from the application configuration and the
process env (environment variables; "env" derives from environment). A short
timeout (time-out as a protective cut-off) is applied to network requests to
avoid blocking local functionality when the network is slow or unavailable.

The sync process is idempotent and safety-oriented: feedback items are assigned
deterministic identifiers and a strictly monotonic checkpoint is recorded only
after the Cloud acknowledges a batch. If no new items exist or local files are
missing, the routine exits cleanly. All logic uses Python's standard library so
that deployments remain portable and free of additional dependencies.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from config import CONFIG
from shared.feedback_client import (
    post_feedback_batch,
    FeedbackClientError,
    FeedbackClientTimeoutError,
)


# Derived storage locations from configuration
USER_DATA_DIR: str = CONFIG["paths"]["user_data_full_path"]
FEEDBACK_DIR: str = os.path.join(
    USER_DATA_DIR, CONFIG["paths"].get("feedback_subdir_name", "feedback")
)
NEGATIVE_FILE: str = os.path.join(
    FEEDBACK_DIR, CONFIG["paths"].get("negative_feedback_filename", "negative_feedback.json")
)
POSITIVE_FILE: str = os.path.join(
    FEEDBACK_DIR, CONFIG["paths"].get("positive_feedback_filename", "positive_feedback.json")
)
SYNC_STATE_PATH: str = os.path.join(USER_DATA_DIR, "feedback_sync_state.json")


def _read_json_array(path: str) -> List[Dict[str, Any]]:
    """
    Read a JSON file expected to contain a list of objects; return an empty list on error.

    This helper opens the given path if it exists and attempts to parse a JSON array.
    If the file is missing, contains invalid JSON, or the root is not a list, it returns
    an empty list. This behavior makes the sync routine robust against partial writes
    and format drift without raising exceptions for common edge cases.

    Args:
        path (str): Absolute or relative filesystem path to a JSON file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries parsed from the file, or an empty list
        if the file is missing or invalid.
    """
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def _now_iso_utc() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _normalize_feedback_item(
    raw: Dict[str, Any],
    label: str,
    idx: int,
) -> Dict[str, Any]:
    """
    Normalize a single feedback item into the unified schema used by the sync API.

    The function tolerates missing fields and fills sensible defaults while preserving
    original content where present. A deterministic `feedback_id` is computed when not
    provided to support idempotency on the Cloud side. Unknown fields are ignored to keep
    the payload compact and predictable.

    Args:
        raw (Dict[str, Any]): The original feedback object read from a local file.
        label (str): Either "positive" or "negative" depending on the source file.
        idx (int): Position index in the source array, used as a salt for deterministic IDs.

    Returns:
        Dict[str, Any]: Normalized feedback object with required keys.
    """
    interaction_id = str(raw.get("interactionId", "")).strip()
    created_at = str(raw.get("created_at", "")).strip() or _now_iso_utc()
    comment = str(raw.get("comment", ""))

    # Compute deterministic 32-hex ID if missing
    fid = str(raw.get("feedback_id", "")).strip()
    if not fid:
        fid = hashlib.sha256(f"{interaction_id}:{created_at or idx}".encode("utf-8")).hexdigest()[:32]
    fid = fid.lower()

    norm_label = "positive" if label == "positive" else "negative"
    score = 1 if norm_label == "positive" else -1

    return {
        "feedback_id": fid,
        "interactionId": interaction_id,
        "label": norm_label,
        "score": score,
        "comment": comment,
        "created_at": created_at,
    }


def _load_local_feedback() -> List[Dict[str, Any]]:
    """
    Load and normalize local feedback items from positive and negative JSON files.

    This helper reads two arrays from the configured positive and negative feedback files
    and converts them into a single list using a consistent schema suitable for the Cloud
    sync API. Items are tagged with a `label` ("positive" or "negative") based on their
    source file and mapped to `score` values (+1 or -1). A `feedback_id` field is ensured
    for each item: if it is missing, a deterministic identifier is generated via
    `sha256(f"{interactionId}:{created_at or idx}").hexdigest()[:32]` to enable safe
    deduplication on the Cloud. Missing `created_at` values are replaced with the current
    UTC timestamp (ISO 8601). The function never raises and returns an empty list if the
    files are missing or invalid, making it safe to call from batch processes.

    Returns:
        List[Dict[str, Any]]: A list of normalized feedback objects ready for syncing.
    """
    pos = _read_json_array(POSITIVE_FILE)
    neg = _read_json_array(NEGATIVE_FILE)

    normalized: List[Dict[str, Any]] = []
    for i, item in enumerate(pos):
        normalized.append(_normalize_feedback_item(item if isinstance(item, dict) else {}, "positive", i))
    for i, item in enumerate(neg):
        normalized.append(_normalize_feedback_item(item if isinstance(item, dict) else {}, "negative", i))
    return normalized


def _read_sync_state() -> Dict[str, Optional[str]]:
    """
    Read the feedback sync state file, defaulting to a fresh state if not present.

    Returns a mapping containing at least the key `last_synced_at`, which is either an ISO
    timestamp string or None when no sync has occurred yet. Any read/parse failure results in
    the default structure to keep the sync routine resilient across runs.
    """
    if not os.path.exists(SYNC_STATE_PATH):
        return {"last_synced_at": None}
    try:
        with open(SYNC_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {"last_synced_at": None}
            return {"last_synced_at": data.get("last_synced_at")}
    except Exception:
        return {"last_synced_at": None}


def _write_sync_state(last_synced_at: str) -> None:
    """
    Persist the feedback sync checkpoint to disk.

    The function overwrites the sync state file with the provided ISO timestamp, representing
    the most recent `created_at` that has been accepted by the Cloud. This checkpoint is used
    on subsequent runs to filter only-new feedback items.
    """
    state = {"last_synced_at": last_synced_at}
    os.makedirs(os.path.dirname(SYNC_STATE_PATH), exist_ok=True)
    with open(SYNC_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _parse_iso(ts: str) -> Optional[_dt.datetime]:
    """Best-effort ISO 8601 parser that supports a trailing 'Z' as UTC."""
    if not ts:
        return None
    try:
        return _dt.datetime.fromisoformat(ts)
    except ValueError:
        try:
            if ts.endswith("Z"):
                return _dt.datetime.fromisoformat(ts[:-1] + "+00:00")
        except Exception:
            return None
    return None


def _filter_new(items: List[Dict[str, Any]], last_synced_at: Optional[str]) -> List[Dict[str, Any]]:
    """
    Return only items with created_at strictly greater than the checkpoint.

    If the checkpoint is None, all items are considered new. Timestamps are compared using
    parsed ISO datetimes; invalid timestamps are treated as new to avoid data loss during
    ingestion. The function does not raise on parse errors and returns a best-effort filter.
    """
    if last_synced_at is None:
        return list(items)

    checkpoint = _parse_iso(last_synced_at)
    if checkpoint is None:
        return list(items)

    newer: List[Dict[str, Any]] = []
    for it in items:
        created_at = _parse_iso(str(it.get("created_at", "")))
        if created_at is None or created_at > checkpoint:
            newer.append(it)
    return newer


def run_feedback_sync() -> Dict[str, Any]:
    """
    Execute a single feedback synchronization run and return a summary.

    The function loads the last sync checkpoint, reads and normalizes local feedback, and
    filters only the new items strictly after the checkpoint. If no new items exist, the
    function returns a summary with zero "sent" and does not change the checkpoint. If
    items exist, it posts the batch to the Cloud endpoint configured via
    `CONFIG["cloud_rag"]["base_url"]` (raises a RuntimeError if missing). Upon a successful
    HTTP 200 response, it advances the checkpoint to the maximum `created_at` among the sent
    items and returns a summary of counts including accepted and duplicates reported by the
    Cloud. Any network timeout raises `TimeoutError`; other failures raise `RuntimeError`.

    Returns:
        Dict[str, Any]: A summary dictionary with keys: `sent`, `accepted`, `duplicates`, `skipped`.
    """
    state = _read_sync_state()
    last_synced_at = state.get("last_synced_at")

    all_items = _load_local_feedback()
    new_items = _filter_new(all_items, last_synced_at)
    skipped = max(0, len(all_items) - len(new_items))
    if not new_items:
        return {"sent": 0, "accepted": 0, "duplicates": 0, "skipped": skipped}

    cloud_cfg = CONFIG.get("cloud_rag", {}) or {}
    base_url = cloud_cfg.get("base_url")
    if not isinstance(base_url, str) or not base_url.strip():
        raise RuntimeError("Missing CONFIG['cloud_rag']['base_url'] for feedback sync")

    try:
        resp = post_feedback_batch(base_url=base_url, items=new_items, timeout_s=5.0)
    except FeedbackClientTimeoutError as exc:
        # Preserve previous public exception surface
        raise TimeoutError(str(exc)) from exc
    except FeedbackClientError as exc:
        raise RuntimeError(str(exc)) from exc
    accepted = int(resp.get("accepted", 0))
    duplicates = int(resp.get("duplicates", 0))

    # Advance checkpoint on success
    newest_ts = max((str(i.get("created_at", "")) for i in new_items), default=None)
    if newest_ts:
        _write_sync_state(newest_ts)

    return {
        "sent": len(new_items),
        "accepted": accepted,
        "duplicates": duplicates,
        "skipped": skipped,
    }


if __name__ == "__main__":
    summary = run_feedback_sync()
    print(json.dumps(summary, indent=2))


