#!/usr/bin/env python3
"""
Edge Server Provider Switcher

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
    template_path = project_root / 'config' / 'templates' / f'edge_{provider}.json'
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
    base_url = config.get('llm', {}).get('base_url', 'unknown')

    print(f"   LLM: {llm_provider}")
    print(f"   Base URL: {base_url}")

    print(f"\n📝 Note: Make sure to set the appropriate API key:")
    if provider == 'nebius':
        print(f"   export NEBIUS_API_KEY=your_key_here")
    else:
        print(f"   export OPENAI_API_KEY=your_key_here")

    print(f"\n🔄 Remember to restart the edge server to apply changes")
    print(f"   poetry run python main.py")


def main():
    parser = argparse.ArgumentParser(
        description="Switch Edge Server LLM provider configuration",
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
