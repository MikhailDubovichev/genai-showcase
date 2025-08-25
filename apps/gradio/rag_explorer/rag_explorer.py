"""
Minimal Gradio RAG Explorer UI for the cloud RAG service.

This app provides a tiny user interface (UI) built with Gradio to interact
with the cloud Retrieval Augmented Generation (RAG) service. Users can type a
question, adjust the topK parameter for retrieval, and optionally provide an
interactionId. The app sends a JSON request to the cloud POST /api/rag/answer,
measures round-trip latency, shows retrieved chunks with scores in a table,
and displays the final validated JSON response. This explorer helps users
understand how the RAG system retrieves relevant context and generates
answers. "LLM" stands for Large Language Model, which is the core engine that
processes natural language and generates human-like responses. "RAG" stands
for Retrieval Augmented Generation, a technique that enhances LLM responses
by retrieving relevant context from a knowledge base before generating answers.
"FAISS" stands for Facebook AI Similarity Search, which is the vector search
library used to find similar documents efficiently. "JSON" stands for JavaScript
Object Notation, a standard format for structured data exchange between systems.
"""

from __future__ import annotations

import json
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Prefer local Gradio dependency; install with: `poetry add gradio`
import gradio as gr

# Ensure local imports resolve to the Gradio app's shared utilities.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.config import (  # noqa: E402
    build_url,
    get_cloud_base_url,
    get_gradio_rag_explorer_url,
    get_gradio_port,
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
    cloud RAG service so that related requests and diagnostics are correlated
    across services, enabling better observability and debugging of the RAG
    pipeline.

    Returns:
        str: A 32-character lowercase hexadecimal UUID string.
    """
    return uuid.uuid4().hex.lower()


def post_answer(
    base_url: str,
    question: str,
    interaction_id: str,
    top_k: int,
    timeout_s: float
) -> Dict[str, Any]:
    """
    Send a JSON RAG answer request to the cloud server using the standard library.

    This helper performs an HTTP POST to the cloud endpoint constructed from the
    provided base URL and the path `/api/rag/answer`. The request body is a JSON
    object with keys `question`, `interactionId`, and `topK`. The function returns
    the parsed JSON object on success, which should contain the generated answer,
    retrieved context chunks with scores, and metadata. If a network or HTTP error
    occurs, the function returns a dictionary with an `error` key containing a
    human-readable explanation, instead of raising. This approach keeps the UI
    logic simple and resilient when the cloud service is unavailable or slow.

    Args:
        base_url (str): Normalized base URL for the cloud RAG API.
        question (str): User question to send for RAG processing.
        interaction_id (str): 32-character hex identifier for tracing.
        top_k (int): Number of top chunks to retrieve from the vector store.
        timeout_s (float): Timeout in seconds for the HTTP request.

    Returns:
        Dict[str, Any]: Parsed JSON on success, or a dictionary with `error`.
    """
    url = build_url(base_url, "/api/rag/answer")
    payload = {
        "question": question,
        "interactionId": interaction_id,
        "topK": top_k,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(raw)
            except Exception as json_exc:  # noqa: BLE001
                return {"error": f"Invalid JSON from cloud: {json_exc}", "raw": raw}
    except urllib.error.HTTPError as http_err:
        try:
            raw = http_err.read().decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        return {"error": f"HTTP {http_err.code}", "raw": raw}
    except urllib.error.URLError as url_err:
        return {"error": f"Network error: {url_err}"}


def build_chunk_table(response_obj: Dict[str, Any]) -> List[List[Any]]:
    """
    Extract and format retrieved chunks with scores from the RAG response.

    This helper processes the response from the cloud RAG service and extracts
    the `content` list, which should contain retrieved chunks with their metadata.
    For each chunk, it creates a row with three columns: the sourceId (identifier
    of the source document), the relevance score rounded to 4 decimal places,
    and the chunk text itself. If the response lacks a valid content list or
    the items are malformed, it returns an empty list. This table helps users
    understand which documents were retrieved, how relevant they were, and what
    specific text was used to generate the answer.

    Args:
        response_obj (Dict[str, Any]): The full response object from the RAG service.

    Returns:
        List[List[Any]]: A list of rows for the dataframe, each containing
                        [sourceId, score, chunk].
    """
    content = response_obj.get("content", [])
    if not isinstance(content, list):
        return []

    table_data = []
    for item in content:
        if isinstance(item, dict):
            source_id = item.get("sourceId", "")
            score = item.get("score", 0.0)
            chunk = item.get("chunk", "")
            try:
                rounded_score = round(float(score), 4)
            except (ValueError, TypeError):
                rounded_score = 0.0
            table_data.append([source_id, rounded_score, chunk])
        else:
            # Skip malformed items
            continue

    return table_data


def _build_ui() -> gr.Blocks:
    """
    Construct the Gradio UI and bind it to the submission handler.

    This function loads the Gradio configuration, computes a normalized base
    URL and timeout, and closes over those values in the `handle_submit`
    function. The UI exposes inputs for question and topK slider, plus outputs
    for raw JSON, extracted message, latency, and a dataframe of retrieved chunks.
    Keeping all wiring within this function makes the module importable without
    side effects and simplifies testing.
    """
    cfg = load_gradio_config()
    base_url = get_cloud_base_url(cfg)
    timeout_s = get_timeout_seconds(cfg)

    def handle_submit(
        question: str,
        top_k: int
    ) -> Tuple[str, str, float, List[List[Any]]]:
        """
        Handle form submission, measure latency, and format the RAG response.

        This function generates an interaction identifier automatically and
        measures wall-clock latency around a single HTTP POST to the cloud RAG
        endpoint. The result is returned as a pretty-printed JSON string,
        the extracted message, the elapsed time in milliseconds, and a table
        of retrieved chunks with their scores. This allows users to see both
        the final answer and the retrieval evidence that supports it.

        Args:
            question (str): The user's question for RAG processing.
            top_k (int): Number of chunks to retrieve (1-10).

        Returns:
            Tuple[str, str, float, List[List[Any]]]: (pretty_json, message, latency_ms, chunk_table)
        """
        iid = generate_interaction_id()
        t0 = time.monotonic()
        obj = post_answer(
            base_url=base_url,
            question=question,
            interaction_id=iid,
            top_k=top_k,
            timeout_s=timeout_s,
        )
        dt_ms = (time.monotonic() - t0) * 1000.0
        try:
            pretty = json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            pretty = str(obj)
        extracted_message = obj.get("message", "No message in response")
        chunk_table = build_chunk_table(obj)
        return pretty, extracted_message, float(max(0.0, dt_ms)), chunk_table

    def handle_reset() -> Tuple[str, str, float, List[List[Any]]]:
        """
        Reset the question field and clear all outputs.

        Returns empty/default values for all outputs to clear the UI.
        """
        return "", "", 0.0, []

    with gr.Blocks(title="RAG Explorer (Cloud)") as demo:
        gr.Markdown("# RAG Explorer (Cloud)")
        with gr.Row():
            question_in = gr.Textbox(label="Question", lines=2)
        with gr.Row():
            send_btn = gr.Button("Send")
            reset_btn = gr.Button("Reset")
        with gr.Row():
            top_k_in = gr.Slider(label="topK", minimum=1, maximum=10, value=3, step=1)
            latency_out = gr.Number(label="Latency (ms)", precision=0)
        with gr.Row():
            json_out = gr.Code(label="Raw JSON")
            message_out = gr.Textbox(label="Message")
        chunk_table = gr.Dataframe(
            headers=["sourceId", "score", "chunk"],
            row_count=(5, "dynamic"),
            label="Retrieved Chunks"
        )

        reset_btn.click(
            fn=handle_reset,
            outputs=[question_in, json_out, message_out, latency_out, chunk_table]
        )
        send_btn.click(
            fn=handle_submit,
            inputs=[question_in, top_k_in],
            outputs=[json_out, message_out, latency_out, chunk_table]
        )

    return demo


if __name__ == "__main__":
    cfg = load_gradio_config()
    gradio_url = get_gradio_rag_explorer_url(cfg)
    port = get_gradio_port(gradio_url)
    print(f"Starting RAG Explorer UI at: {gradio_url}")
    _build_ui().launch(share=False, server_port=port, server_name="0.0.0.0")
