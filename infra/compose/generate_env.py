#!/usr/bin/env python3
"""
Generate .env file from application config files for Docker Compose.

This script reads the configuration from the various app config.json files
and generates a .env file with the appropriate environment variables for
docker-compose.yml. This makes the Docker setup truly dynamic and in sync
with the application configurations.

Usage:
    python generate_env.py [--output .env]
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, Any


def load_json_config(config_path: Path) -> Dict[str, Any]:
    """Load JSON configuration file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON in {config_path}: {e}")
        return {}


def extract_port_from_url(url: str) -> str:
    """Extract port number from a URL string."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return str(parsed.port) if parsed.port else ""
    except Exception:
        return ""


def generate_env_content() -> str:
    """Generate .env file content from app configurations."""
    base_dir = Path(__file__).resolve().parent.parent.parent  # Project root

    # Load configurations
    edge_config = load_json_config(base_dir / "apps" / "edge-server" / "config" / "config.json")
    gradio_config = load_json_config(base_dir / "apps" / "gradio" / "config" / "config.json")

    env_vars = []

    # Extract ports from configurations
    if gradio_config:
        # Edge server port (from gradio config's edge_api_base_url)
        edge_api_url = gradio_config.get("edge_api_base_url", "")
        if edge_api_url:
            port = extract_port_from_url(edge_api_url)
            if port:
                env_vars.append(f"EDGE_SERVER_PORT={port}")

        # Cloud RAG port (from gradio config's cloud_rag_base_url)
        cloud_rag_url = gradio_config.get("cloud_rag_base_url", "")
        if cloud_rag_url:
            port = extract_port_from_url(cloud_rag_url)
            if port:
                env_vars.append(f"CLOUD_RAG_PORT={port}")

        # Gradio UI ports
        gradio_edge_chat_url = gradio_config.get("gradio_edge_chat_url", "")
        if gradio_edge_chat_url:
            port = extract_port_from_url(gradio_edge_chat_url)
            if port:
                env_vars.append(f"GRADIO_CHAT_PORT={port}")

        gradio_rag_explorer_url = gradio_config.get("gradio_rag_explorer_url", "")
        if gradio_rag_explorer_url:
            port = extract_port_from_url(gradio_rag_explorer_url)
            if port:
                env_vars.append(f"GRADIO_RAG_EXPLORER_PORT={port}")

    # Add comment header
    content = [
        "# Auto-generated .env file for Docker Compose",
        "# Generated from application config files",
        "# Do not edit manually - use generate_env.py instead",
        ""
    ]

    content.extend(env_vars)

    return "\n".join(content)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate .env file from app configs")
    parser.add_argument("--output", "-o", default=".env", help="Output .env file path")
    args = parser.parse_args()

    # Generate content
    env_content = generate_env_content()

    # Write to file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        f.write(env_content)

    print(f"Generated {output_path} with {len(env_content.splitlines())} lines")
    print(f"Environment variables: {[line.split('=')[0] for line in env_content.splitlines() if line and not line.startswith('#')]}")

    # Show current config values
    print("\nCurrent configuration:")
    for line in env_content.splitlines():
        if line and not line.startswith('#'):
            print(f"  {line}")


if __name__ == "__main__":
    main()
