# Configuration Management

This directory contains configuration files and templates for the Cloud RAG service.

## Structure

```
config/
├── config.json              # Active configuration (DO NOT EDIT DIRECTLY)
├── templates/               # Configuration templates
│   ├── config.nebius.json   # Nebius provider configuration
│   └── config.openai.json   # OpenAI provider configuration
└── README.md               # This file
```

## Provider Switching

Use the switching script to easily switch between providers:

### Switch to Nebius
```bash
python scripts/switch_provider.py nebius
```

### Switch to OpenAI  
```bash
python scripts/switch_provider.py openai
```

## Manual Switching

If you prefer to switch manually:

```bash
# To Nebius
cp config/templates/config.nebius.json config/config.json

# To OpenAI
cp config/templates/config.openai.json config/config.json
```

## After Switching

1. **Set the appropriate API key:**
   - **Nebius:** `export NEBIUS_API_KEY=your_key_here`
   - **OpenAI:** `export OPENAI_API_KEY=your_key_here`

2. **If switching embeddings, re-seed the FAISS index:**
   ```bash
   rm -rf faiss_index
   poetry run python scripts/seed_index.py
   ```

## Configuration Differences

### Nebius Configuration
- **LLM Model:** Qwen/Qwen3-235B-A22B-Instruct-2507
- **Embeddings:** Qwen/Qwen3-Embedding-8B
- **Base URL:** https://api.studio.nebius.ai/v1
- **API Key:** NEBIUS_API_KEY

### OpenAI Configuration
- **LLM Model:** gpt-4
- **Embeddings:** text-embedding-3-small
- **Base URL:** https://api.openai.com/v1
- **API Key:** OPENAI_API_KEY

## Best Practices

1. **Never edit `config.json` directly** - use templates and switching scripts
2. **Always re-seed when switching embeddings** to avoid dimension mismatches
3. **Test with both providers** to ensure compatibility
4. **Keep API keys in environment variables** (never in config files)
5. **Document any custom configurations** you create

## Adding New Providers

1. Create a new template: `config/templates/config.newprovider.json`
2. Add the provider implementation in `providers/newprovider_*.py`
3. Update the factory in `providers/factory.py`
4. Add the provider to the switching script
5. Update this README
