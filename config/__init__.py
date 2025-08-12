import os
import json
from pathlib import Path
from dotenv import load_dotenv
from .logging_config import setup_app_logging
import logging

# Load environment variables from .env file
load_dotenv()

# Get the config directory path (where this file is located)
CONFIG_DIR = Path(__file__).parent

# <<< Define PROJECT_ROOT based on CONFIG_DIR >>>
PROJECT_ROOT = CONFIG_DIR.parent

# Load configuration from config.json
config_path = CONFIG_DIR / 'config.json'
with open(config_path, 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)

# <<< Add PROJECT_ROOT to CONFIG and derive full paths >>>
# We programatically add PROJECT_ROOT to CONFIG Python dictionary.
# CONFIG is an in-memory Python dictionary that is used to store the configuration.
# This is a runtime representation of our configuration (config.json).
CONFIG['project_root'] = str(PROJECT_ROOT) # Store as string for easier use

# Ensure 'paths' key exists in CONFIG, defaulting to an empty dict if not
if 'paths' not in CONFIG:
    CONFIG['paths'] = {}

# Get base and subdir names from CONFIG, using .get for safety, with fallbacks.
user_data_base_name = CONFIG.get('paths', {}).get('user_data_base_dir_name', 'user_data')
contexts_subdir_name = CONFIG.get('paths', {}).get('contexts_subdir_name', 'contexts')

# Calculate and store full absolute paths in the 'paths' dictionary
CONFIG['paths']['user_data_full_path'] = str(PROJECT_ROOT / user_data_base_name)
CONFIG['paths']['contexts_full_path'] = str(PROJECT_ROOT / user_data_base_name / contexts_subdir_name)

# --- System Prompt Loading ---
# Load system prompts for different pipelines

# Load device control system prompt
device_control_prompt_path = CONFIG_DIR / 'device_control_system_prompt.txt'
try:
    with open(device_control_prompt_path, 'r', encoding='utf-8') as f:
        CONFIG['device_control_message'] = f.read().strip()
except FileNotFoundError:
    raise FileNotFoundError(
        f"Device control system prompt file not found: {device_control_prompt_path}\n"
        f"Please ensure device_control_system_prompt.txt exists in the config directory."
    )

# Load energy efficiency system prompt
energy_efficiency_prompt_path = CONFIG_DIR / 'energy_efficiency_system_prompt.txt'
try:
    with open(energy_efficiency_prompt_path, 'r', encoding='utf-8') as f:
        CONFIG['energy_efficiency_message'] = f.read().strip()
except FileNotFoundError:
    # Fallback to main system prompt if energy efficiency prompt missing
    CONFIG['energy_efficiency_message'] = CONFIG['device_control_message']

# Environment variables
ENV = {
    'LLM_API_KEY': os.getenv('NEBIUS_API_KEY'),  # Using old env var name for backward compatibility
}

def validate_config():
    """Validate that all required environment variables and configuration settings are present.

    This setup is integrator-agnostic. Only LLM configuration is strictly required by
    this module. Provider/integrator configuration is optional and handled by the
    provider layer.
    """
    # Check environment variables
    missing_env_vars = [key for key, value in ENV.items() if value is None]
    if missing_env_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_env_vars)}\n"
            f"Please check your .env file."
        )

    # Check if required services are configured
    required_services = ['llm']  # integrator config is optional
    for service in required_services:
        if service not in CONFIG:
            raise ValueError(f"Missing configuration for service: {service}")
            
    # Check if all required LLM models are configured
    required_models = ['classification', 'device_control', 'energy_efficiency']
    for model in required_models:
        if model not in CONFIG['llm']['models']:
            raise ValueError(f"Missing configuration for LLM model: {model}")

# Validate configuration on module import
validate_config()

# --- Helper function to get config value from CONFIG or environment variable ---
def get_config_value(json_keys: list, env_var_name: str, default_value: any = None):
    """
    Retrieves a configuration value.
    Priority:
    1. Environment variable (if env_var_name is provided and variable is set).
    2. Value from CONFIG dictionary (using json_keys).
    3. default_value.
    """
    # Try environment variable first
    if env_var_name:
        env_value = os.getenv(env_var_name)
        if env_value is not None:
            # Attempt to match type of default_value if it's int or bool
            if isinstance(default_value, bool):
                if env_value.lower() == 'true': return True
                if env_value.lower() == 'false': return False
            elif isinstance(default_value, int):
                try:
                    return int(env_value)
                except ValueError:
                    pass # Fall through to JSON or default if not a valid int
            return env_value # Return as string if no type match or not int/bool

    # Try from CONFIG dictionary (loaded from JSON)
    current_level = CONFIG
    try:
        for key in json_keys:
            current_level = current_level[key]
        # If found, attempt to match type of default_value for consistency if it's int or bool
        if isinstance(default_value, bool) and isinstance(current_level, bool):
            return current_level
        if isinstance(default_value, int) and isinstance(current_level, int):
            return current_level
        # If it's a string value from JSON, return it.
        # Or if types don't match typical bool/int, return what's in JSON.
        if isinstance(current_level, (str, int, bool, float, list, dict)): # Check if it's a typical JSON type
             return current_level
    except (KeyError, TypeError):
        pass # Key not found or CONFIG structure not as expected, fall through to default

    # Fallback to default value
    return default_value

# --- Logging Configuration ---
# Defaults for logging config are also in logging_config.py's setup_app_logging function's signature
# or can be specified in config.json. Environment variables take precedence.
CONFIG['logging'] = {
    'level': get_config_value(['logging', 'level'], 'LOG_LEVEL', 'INFO'),
    'file_path': get_config_value(['logging', 'file_path'], 'LOG_FILE_PATH', 'logs/ai_assistant.log'),
    'max_bytes': get_config_value(['logging', 'max_bytes'], 'LOG_MAX_BYTES', 5*1024*1024), # 5MB
    'backup_count': get_config_value(['logging', 'backup_count'], 'LOG_BACKUP_COUNT', 3),
    'format': get_config_value(
        ['logging', 'format'],
        'LOG_FORMAT',
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
    ),
    'date_format': get_config_value(
        ['logging', 'date_format'],
        'LOG_DATE_FORMAT',
        '%Y-%m-%d %H:%M:%S'
    )
}

# --- Setup Application Logging ---
# Set up logging for the entire application.
setup_app_logging(config=CONFIG.get('logging')) # Pass the logging-specific part of CONFIG

# each module also should have its own logger, so we can use the same logging config for all modules.
# here we are getting a logger for the config module.
# technically getLogger uses settings already set up by setup_app_logging and applied globally (to the root logger)
config_init_logger = logging.getLogger(__name__) # Get logger for this module AFTER logging is setup
config_init_logger.info("[config_init] Logging initialized from config/__init__.py using setup_app_logging.\n") 