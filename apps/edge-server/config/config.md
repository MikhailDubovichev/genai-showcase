Here I add several comments on config.json file. I can't add comments directly to the json schema - it isn't allowed

# Smart Home Integrator Configuration

## API Configuration
- `base_url`: The base URL for the Smart Home Integrator HTTP API. This may point to a sandbox or staging host during development.
- `service_location_id`: Fallback service location ID used for testing. The actual location ID can be provided in API calls.
- `timeout`: API request timeout in seconds.

### Endpoints
- `smart_devices`: Endpoint for retrieving the list of devices available at a service location.
- `automations`: Endpoint for automation resources (if used by a real integrator).
- `device_action`: Endpoint for executing immediate actions (for example, "on" or "off") on a specific device.

Note: Pricing-related endpoints (for example, dynamic tariffs or spot prices) are not used by the current mock integrator and can be ignored for the public showcase.

## Example Hosts

Depending on your environment, the integrator may expose different base URLs (for example, staging vs. production). Update `base_url` and `host` accordingly in `config.json`.

---

### Logging Configuration (`logging`)

This section configures the application-wide logging behavior. Logging is crucial for monitoring the application, debugging issues, and understanding its runtime behavior.

Settings are typically defined in `config.json` and can be overridden by environment variables.

- **`level`**: The minimum severity level of log messages that will be processed.
  - JSON Key: `logging.level`
  - Env Var: `LOG_LEVEL`
  - Values: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
  - Default: `INFO` (if not set in JSON or env var).
  - Example: `"INFO"` in `config.json` or `LOG_LEVEL=DEBUG` in `.env`.

- **`file_path`**: The path to the log file where messages will be written. The directory for the log file (for example, `logs/`) should exist.
  - JSON Key: `logging.file_path`
  - Env Var: `LOG_FILE_PATH`
  - Default: `logs/ai_assistant.log`.
  - Example: `"logs/my_app.log"` in `config.json` or `LOG_FILE_PATH=/var/log/app/service.log` in `.env`.

- **`max_bytes`**: The maximum size (in bytes) that a log file can reach before it is rotated. Once this size is exceeded, the current log file is renamed (backed up), and a new log file is started.
  - JSON Key: `logging.max_bytes`
  - Env Var: `LOG_MAX_BYTES`
  - Default: `5242880` (5MB).
  - Example: `5242880` in `config.json` or `LOG_MAX_BYTES=10485760` (10MB) in `.env`.

- **`backup_count`**: The number of old (rotated) log files to keep. Once this number is reached, the oldest backup file is deleted when a new rotation occurs.
  - JSON Key: `logging.backup_count`
  - Env Var: `LOG_BACKUP_COUNT`
  - Default: `3`.
  - Example: `5` in `config.json` or `LOG_BACKUP_COUNT=10` in `.env`.

- **`format`**: A string defining the format of the log messages. It uses standard Python `logging` module format codes.
  - JSON Key: `logging.format`
  - Env Var: `LOG_FORMAT`
  - Default: `%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s`
    - `%(asctime)s`: Time the log record was created.
    - `%(name)s`: Name of the logger (usually the module name).
    - `%(levelname)s`: Text logging level for the message (for example, INFO, DEBUG).
    - `%(module)s`: Module name.
    - `%(lineno)d`: Source line number where the logging call was issued.
    - `%(message)s`: The logged message.
  - Example: `"%(asctime)s [%(levelname)s] %(message)s"` in `config.json`.

- **`date_format`**: A string defining the format for the `asctime` part of the log message. It uses standard Python `time.strftime()` format codes.
  - JSON Key: `logging.date_format`
  - Env Var: `LOG_DATE_FORMAT`
  - Default: `%Y-%m-%d %H:%M:%S`.
  - Example: `"%m/%d/%Y %I:%M:%S %p"` in `config.json`.

## Path Configuration (`paths`)

This section in `config.json` defines base names for key directories used by the application. The application then programmatically determines the project's root directory and constructs full absolute paths to these data directories. These constructed paths, along with the project root, are added to the global `CONFIG` object for internal use.

- **`user_data_base_dir_name`** (in `config.json`)
  - Description: The name of the base directory (relative to project root) for storing all user-related data.
  - JSON Key: `paths.user_data_base_dir_name`
  - Default (if not in `config.json`, fallback in `config/__init__.py`): "user_data"
  - Example: `"user_data_custom_name"`

- **`contexts_subdir_name`** (in `config.json`)
  - Description: The name of the subdirectory (relative to `user_data_base_dir_name`) where LLM context files (like device lists) are stored.
  - JSON Key: `paths.contexts_subdir_name`
  - Default (if not in `config.json`, fallback in `config/__init__.py`): "contexts"
  - Example: `"llm_contexts"`

--- 
**Programmatically Added to `CONFIG` (in `config/__init__.py`):**

- **`CONFIG['project_root']`**
  - Description: The absolute path to the project's root directory.
  - Source: Calculated in `config/__init__.py` based on its own file location.
  - Usage: Used internally by the application to construct other absolute paths.

- **`CONFIG['paths']['user_data_full_path']`**
  - Description: The full absolute path to the user data directory.
  - Source: Calculated in `config/__init__.py` using `CONFIG['project_root']` and `CONFIG['paths']['user_data_base_dir_name']`.
  - Usage: Used internally by the application to access the user data directory.

- **`CONFIG['paths']['contexts_full_path']`**
  - Description: The full absolute path to the LLM contexts directory.
  - Source: Calculated in `config/__init__.py` using `CONFIG['paths']['user_data_full_path']` and `CONFIG['paths']['contexts_subdir_name']` (or their fallbacks).
  - Usage: Used internally by the application to access the contexts directory (for example, in `api/context.py`).

## LLM Error Messages (`llm_error_messages`)

This section stores predefined messages for various error scenarios encountered during LLM interactions. This allows for easy customization and localization of user-facing error messages.

- **`json_parse_fallback`**: The fallback message sent to the user if the application fails to parse the LLM's response (for example, if the LLM returns malformed JSON).
  - JSON Key: `llm_error_messages.json_parse_fallback`
  - Example: `"I'm having a little trouble understanding the response from my core intelligence. Please try again."`