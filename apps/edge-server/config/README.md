# Edge Server Configuration Management

This directory contains configuration templates and utilities for managing LLM provider settings in the Edge Server.

## Directory Structure

```
config/
├── config.json              # Active configuration (DO NOT EDIT DIRECTLY)
├── templates/               # Provider configuration templates
│   ├── edge_nebius.json     # Nebius provider configuration
│   └── edge_openai.json     # OpenAI provider configuration
├── README.md               # This file (configuration management docs)
└── config.md               # General configuration documentation
```

## Provider Templates

The templates contain **only** the configuration keys that need to be overridden when switching providers. This ensures:

- **Minimal changes**: Only provider-specific settings are modified
- **Safety**: All other configuration remains intact
- **Consistency**: Same base configuration with different provider settings

### Template Contents

#### Nebius Template (`edge_nebius.json`)
```json
{
  "llm": {
    "provider": "nebius",
    "base_url": "https://api.studio.nebius.com/v1/"
  }
}
```

#### OpenAI Template (`edge_openai.json`)
```json
{
  "llm": {
    "provider": "openai",
    "base_url": "https://api.openai.com/v1/"
  }
}
```

## Provider Switching Script

Use the `scripts/switch_provider.py` script to switch between providers:

### Basic Usage

```bash
# Switch to Nebius
python scripts/switch_provider.py nebius

# Switch to OpenAI
python scripts/switch_provider.py openai
```

### What the Script Does

1. **Copies template** from `config/templates/edge_{provider}.json` to `config/config.json`
2. **Shows summary** of the provider and base URL that was set
3. **Provides reminders** about API keys and server restart

## Safety Features

- **Template validation**: Checks template files exist before proceeding
- **Simple copy**: Directly replaces config with template
- **Clear output**: Shows exactly what provider and base URL was set

## After Switching Providers

### 1. Set API Key
```bash
# For Nebius
export NEBIUS_API_KEY=your_nebius_key_here

# For OpenAI
export OPENAI_API_KEY=your_openai_key_here
```

### 2. Restart Edge Server
The configuration changes require a server restart to take effect:
```bash
# Stop current server (Ctrl+C if running in terminal)
# Then restart
poetry run python main.py
```

## Important Notes

### Provider Wiring
- **This step only updates configuration**
- **Provider implementation wiring happens in M11 Step 5/6**
- The `llm_cloud/provider.py` module will use these config values in later steps

### Model Names
- **Templates don't change model names** (classification, device_control, energy_efficiency)
- Model naming strategy will be handled in future steps if needed
- Current model names remain unchanged when switching providers

### Secrets
- **API keys are NEVER stored in config files**
- Always use environment variables: `NEBIUS_API_KEY` or `OPENAI_API_KEY`
- Templates contain only non-sensitive configuration

## Troubleshooting

### "Template not found" Error
```bash
# Check template files exist
ls config/templates/
# Should show: edge_nebius.json edge_openai.json
```

### "Permission denied" Error
```bash
# Ensure write permissions on config directory
chmod 755 config/
chmod 644 config/config.json
```

### Configuration Not Applied
```bash
# Verify changes were applied
cat config/config.json | grep '"provider"'
# Should show the new provider value

# Restart server to apply changes
poetry run python main.py
```

## Examples

### Complete Workflow
```bash
# 1. Switch to OpenAI
python scripts/switch_provider.py openai

# 2. Set API key
export OPENAI_API_KEY=sk-your-key-here

# 3. Restart server
poetry run python main.py
```

### Switching Back to Nebius
```bash
python scripts/switch_provider.py nebius
export NEBIUS_API_KEY=your-nebius-key
poetry run python main.py
```

## Manual Switching

If you prefer to switch manually:
```bash
# To Nebius
cp config/templates/edge_nebius.json config/config.json

# To OpenAI
cp config/templates/edge_openai.json config/config.json
```
