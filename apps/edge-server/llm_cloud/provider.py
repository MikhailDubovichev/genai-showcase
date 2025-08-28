"""
provider.py – External LLM client with provider validation. Build and return a configured Nebius/OpenAI client
---------------------------------------------------------------------------------------------------------
In the overall data-flow this file sits at the infrastructure layer.
It is the single place where we talk to the external LLM platform (Nebius' OpenAI-compatible endpoint).

Why a *provider* module?
• Keeps third-party SDK initialisation separate from business logic.
• Offers a tiny, easily mockable `get_client()` function instead of a
  global singleton. Tests can monkey-patch this function or inject a fake
  client without importing heavy objects.
• Other internal modules (`chat.py`) simply ask for a client; they do
  not need to know about base URLs or API keys.
• Now includes provider validation to ensure required environment variables
  are present before attempting to build clients, following the same pattern
  as the cloud service (M11 Step 1).

The validation happens at client creation time (not import time) to avoid
failing imports during testing or when the module is loaded without running
the full application. This keeps the module importable for testing while
still enforcing configuration requirements at runtime.
"""

import logging
import os
from typing import Dict, List, Tuple

from openai import OpenAI
from config import CONFIG, ENV

logger = logging.getLogger(__name__)

# NOTE: We deliberately wrap client creation in a **function** instead of a
# module-level global (creation of global instance of the client, CLIENT = OpenAI(...) when module is imported).
# That avoids import-time side effects and lets the caller decide when the client should be built.
# This approach allows to avoid potential side effects, so now:
# - client is not created when module is imported, but only when caller asks for it (if the client is created but isn't used, it consumes resources).
# - for testing, we can easily inject a fake client.
# - provider validation happens at client creation time, not import time.


def require_any_env(var_names: List[str]) -> Tuple[str, str]:
    """
    Check that at least one of the specified environment variables is present and non-empty.

    This helper function provides flexible environment variable validation by allowing
    multiple acceptable variable names (e.g., both LLM_API_KEY and NEBIUS_API_KEY for
    backward compatibility). It returns the first valid variable found, which is useful
    for logging which specific variable was used without exposing sensitive values.

    The function never logs or returns the actual secret values, only the variable name
    that was found to contain a valid secret. This maintains security while providing
    useful diagnostic information for troubleshooting configuration issues.

    Args:
        var_names (List[str]): List of environment variable names to check, in order of preference.
            Examples: ["LLM_API_KEY", "NEBIUS_API_KEY"] or ["OPENAI_API_KEY"]

    Returns:
        Tuple[str, str]: A tuple of (selected_var_name, value) where selected_var_name is the
            name of the environment variable that was found, and value is the secret content.

    Raises:
        RuntimeError: If none of the specified environment variables are present or are empty.
            The error message includes all attempted variable names to help with debugging.

    Example:
        >>> var_name, secret = require_any_env(["LLM_API_KEY", "NEBIUS_API_KEY"])
        >>> print(f"Using {var_name} for authentication")
        Using LLM_API_KEY for authentication
    """
    for var_name in var_names:
        value = os.getenv(var_name, "")
        if value:
            return var_name, value

    # None of the variables were found
    var_list = ", ".join(var_names)
    raise RuntimeError(
        f"Missing required environment variable. Set one of: {var_list}"
    )


def validate_env_for_provider(config: Dict) -> None:
    """
    Validate that required environment variables are present for the configured LLM provider.

    This function inspects the provider configuration and ensures that the appropriate
    API key environment variables are available before attempting to build an LLM client.
    It supports both Nebius and OpenAI providers with flexible variable name acceptance
    to maintain compatibility between edge and cloud services.

    For security, the function never logs or exposes secret values, only the names of
    the environment variables that were found. This allows for safe logging while still
    providing useful diagnostic information when configuration issues occur.

    Args:
        config (Dict): The configuration dictionary, expected to contain an 'llm' section
            with a 'provider' key specifying either 'nebius' or 'openai'.

    Raises:
        ValueError: If an unsupported provider is configured. Currently supports only
            'nebius' and 'openai' (case-insensitive).
        RuntimeError: If the required environment variables for the selected provider
            are missing or empty, with a clear message indicating which variables to set.

    Example:
        >>> config = {"llm": {"provider": "nebius"}}
        >>> validate_env_for_provider(config)
        # Raises RuntimeError if neither LLM_API_KEY nor NEBIUS_API_KEY are set

        >>> config = {"llm": {"provider": "openai"}}
        >>> validate_env_for_provider(config)
        # Raises RuntimeError if OPENAI_API_KEY is not set
    """
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "nebius").strip().lower()

    logger.info("LLM provider selected: %s", provider)

    if provider == "nebius":
        # Accept either edge convention (LLM_API_KEY) or cloud convention (NEBIUS_API_KEY)
        selected_var, _ = require_any_env(["LLM_API_KEY", "NEBIUS_API_KEY"])
        logger.info("Using environment variable: %s", selected_var)
    elif provider == "openai":
        selected_var, _ = require_any_env(["OPENAI_API_KEY"])
        logger.info("Using environment variable: %s", selected_var)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def get_client() -> OpenAI:
    """
    Build and return a configured OpenAI client with provider validation.

    This function first validates that the required environment variables are present
    for the configured provider before attempting to build the client. This ensures
    that configuration errors are caught early with clear error messages, rather than
    failing later during actual API calls.

    The provider encapsulates all environment and configuration lookups (base URL, API key, and timeouts)
    so that callers do not need to deal with SDK initialization details. Returning a fresh client instead of
    a global singleton makes tests simpler (the function can be monkey‑patched) and avoids import‑time side
    effects in production.

    The validation step follows the same pattern as the cloud service (M11 Step 1), ensuring
    consistency across the codebase while maintaining the edge service's existing behavior.

    Returns:
        OpenAI: A ready‑to‑use client configured using `CONFIG` and `ENV` values.

    Raises:
        RuntimeError: If required environment variables are missing (via validation).
        ValueError: If an unsupported provider is configured.
    """
    # Validate environment before building client
    validate_env_for_provider(CONFIG)

    return OpenAI(
        base_url=CONFIG["llm"]["base_url"],
        api_key=ENV["LLM_API_KEY"],  # Keep existing behavior for now
        timeout=CONFIG["llm"].get("timeout", 30),  # seconds – explicit is better than implicit
    )


 