"""
Minimal Gradio Chat UI for the edge FastAPI server.

This app provides a tiny user interface (UI) built with Gradio to interact
with the edge server's prompt endpoint. Users can type a free-form message,
optionally supply an `interactionId`, and submit. The app sends a JSON request
to the edge `POST /api/prompt`, measures the round-trip latency, and renders
the raw JSON response for transparency and debugging. "LLM" stands for Large
Language Model, which is the underlying engine used by the edge/cloud services
to process natural language. "Latency" refers to the total time taken from the
moment the request is sent until the response is received. The request and
response bodies are encoded as "JSON" (JavaScript Object Notation), a common
standard for structured data. This app is intentionally dependency-light and
uses Python's standard library (`urllib`) for HTTP.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request

# Prefer local Gradio dependency; install with: `poetry add gradio`
import gradio as gr

# Ensure local imports resolve to the Gradio app's shared utilities.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.config import (  # noqa: E402
    build_url,
    get_edge_base_url,
    get_timeout_seconds,
    load_gradio_config,
)


def generate_interaction_id() -> str:
    """
    Generate a 32-character hexadecimal interaction identifier.

    This function returns a lowercase, dash-free UUID string using
    `uuid.uuid4().hex.lower()`. The 32-character hex format is compatible with
    systems that require a compact identifier (for example, some tracing tools)
    and avoids ambiguity from hyphens. The identifier can be supplied to the
    edge server so that related requests and diagnostics are correlated across
    services.

    Returns:
        str: A 32-character lowercase hexadecimal UUID string.
    """
    return uuid.uuid4().hex.lower()


def post_prompt(base_url: str, message: str, timeout_s: float) -> Dict[str, Any]:
    """
    Send a request to the edge server using query parameters (no JSON body).

    The edge /api/prompt endpoint expects inputs as query parameters. This
    function builds the URL with message, token, location_id, and interactionId
    and performs a POST with an empty body. On success, it parses and returns the
    JSON response. On network/HTTP errors, it returns a dictionary with an 'error'
    key for display in the UI.

    Args:
        base_url (str): Normalized base URL for the edge API.
        message (str): User message to send.
        timeout_s (float): Timeout in seconds for the HTTP request.

    Returns:
        Dict[str, Any]: Parsed JSON on success, or a dictionary with 'error'.
    """
    interaction_id = generate_interaction_id()
    token = "dummy"
    location_id = "1000299"
    base = build_url(base_url, "/api/prompt")
    query = {
        "message": message,
        "token": token,
        "location_id": location_id,
        "interactionId": interaction_id,
    }
    url = f"{base}?{urllib.parse.urlencode(query)}"
    req = urllib.request.Request(url=url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(raw)
            except Exception as json_exc:  # noqa: BLE001
                return {"error": f"Invalid JSON from edge: {json_exc}", "raw": raw}
    except urllib.error.HTTPError as http_err:
        try:
            raw = http_err.read().decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        return {"error": f"HTTP {http_err.code}", "raw": raw}
    except urllib.error.URLError as url_err:
        return {"error": f"Network error: {url_err}"}


def _build_ui() -> gr.Blocks:
    """
    Construct the Gradio UI and bind it to the submission handler.

    This function loads the Gradio configuration, computes a normalized base
    URL and timeout, and defines the UI with a simplified layout: message input,
    reset button, latency, send button, and outputs for raw JSON and extracted message.
    """
    cfg = load_gradio_config()
    base_url = get_edge_base_url(cfg)
    timeout_s = get_timeout_seconds(cfg)

    def handle_submit(message: str) -> Tuple[str, str, float]:
        """
        Handle form submission, measure latency, and format the JSON response.

        Returns the pretty-printed JSON, the extracted message, and latency.
        """
        t0 = time.monotonic()
        obj = post_prompt(base_url=base_url, message=message, timeout_s=timeout_s)
        dt_ms = (time.monotonic() - t0) * 1000.0
        try:
            pretty = json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            pretty = str(obj)
        extracted_message = obj.get("message", "No message in response")
        return pretty, extracted_message, float(max(0.0, dt_ms))

    def handle_reset() -> Tuple[str, str, str, float]:
        """
        Reset the message field and clear outputs.
        """
        return "", "", "", 0.0

    with gr.Blocks(title="Edge Chat (FastAPI)") as demo:
        gr.Markdown("# Edge Chat (FastAPI)")
        with gr.Row():
            message_in = gr.Textbox(label="Message", lines=3)
        with gr.Row():
            send_btn = gr.Button("Send")
            reset_btn = gr.Button("Reset")
            latency_out = gr.Number(label="Latency (ms)", precision=0)
        with gr.Row():
            json_out = gr.Code(label="Raw JSON")
            message_out = gr.Textbox(label="Response Message")

        reset_btn.click(fn=handle_reset, outputs=[message_in, json_out, message_out, latency_out])
        send_btn.click(fn=handle_submit, inputs=[message_in], outputs=[json_out, message_out, latency_out])

    return demo


if __name__ == "__main__":
    _build_ui().launch(share=False, server_port=7860, server_name="0.0.0.0")