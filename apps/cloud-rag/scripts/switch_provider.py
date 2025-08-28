#!/usr/bin/env python3
"""
Provider switching script for Cloud RAG service.

Usage:
    python scripts/switch_provider.py nebius
    python scripts/switch_provider.py openai

This script copies the appropriate template config to config.json,
allowing easy switching between providers without manual file editing.
"""

import argparse
import shutil
from pathlib import Path


def switch_provider(provider: str) -> None:
    """
    Switch the active configuration to the specified provider.
    
    Args:
        provider: Either 'nebius' or 'openai'
    
    Raises:
        FileNotFoundError: If the template config doesn't exist
        ValueError: If provider is not supported
    """
    if provider not in ['nebius', 'openai']:
        raise ValueError(f"Unsupported provider: {provider}. Use 'nebius' or 'openai'")
    
    # Paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    template_path = project_root / 'config' / 'templates' / f'config.{provider}.json'
    config_path = project_root / 'config' / 'config.json'
    
    # Check if template exists
    if not template_path.exists():
        raise FileNotFoundError(f"Template config not found: {template_path}")
    
    # Copy template to active config
    shutil.copy2(template_path, config_path)
    
    print(f"✅ Switched to {provider.upper()} provider")
    print(f"   Template: {template_path}")
    print(f"   Active config: {config_path}")
    
    # Show provider info
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    llm_provider = config.get('llm', {}).get('provider', 'unknown')
    emb_provider = config.get('embeddings', {}).get('provider', 'unknown')
    llm_model = config.get('llm', {}).get('model', 'unknown')
    emb_model = config.get('embeddings', {}).get('name', 'unknown')
    
    print(f"   LLM: {llm_provider} ({llm_model})")
    print(f"   Embeddings: {emb_provider} ({emb_model})")
    
    print(f"\n📝 Note: Make sure to set the appropriate API key:")
    if provider == 'nebius':
        print(f"   export NEBIUS_API_KEY=your_key_here")
    else:
        print(f"   export OPENAI_API_KEY=your_key_here")
    
    print(f"\n🔄 If switching embeddings, remember to re-seed:")
    print(f"   rm -rf faiss_index")
    print(f"   poetry run python scripts/seed_index.py")


def main():
    parser = argparse.ArgumentParser(
        description="Switch Cloud RAG provider configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/switch_provider.py nebius
  python scripts/switch_provider.py openai
        """
    )
    parser.add_argument(
        'provider',
        choices=['nebius', 'openai'],
        help='Provider to switch to'
    )
    
    args = parser.parse_args()
    switch_provider(args.provider)


if __name__ == '__main__':
    main()
