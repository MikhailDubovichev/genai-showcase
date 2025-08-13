"""
Centralized logging configuration for the AI Assistant application.

This module provides a function to set up application-wide logging,
including formatting, log levels, and handlers for console and file output.
"""

import logging
import logging.handlers # Required for RotatingFileHandler
import sys # To ensure we can always output to stdout for console
import json
from typing import Optional

class StructuredLogFormatter(logging.Formatter):
    """
    Custom formatter that adds structured data to log messages.
    
    Features:
    - Includes interaction_id if present in extra fields
    - Includes pipeline_name if present in extra fields
    - Formats tool calls and their arguments
    - Preserves standard log fields (timestamp, level, etc.)
    """
    
    def format(self, record):
        # Create base log structure
        log_data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()
        }
        
        # Add interaction_id if present
        if hasattr(record, 'interaction_id'):
            log_data['interaction_id'] = record.interaction_id
            
        # Add pipeline_name if present
        if hasattr(record, 'pipeline_name'):
            log_data['pipeline_name'] = record.pipeline_name
            
        # Add any extra fields from record
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
            
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

# Default format now includes placeholders for interaction_id and pipeline
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(interaction_id)s] - [%(pipeline_name)s] - %(message)s'
DEFAULT_LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the structured formatter.
    
    Args:
        name (str): Logger name (usually __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Add null values for our custom fields to avoid KeyError
    logger = logging.LoggerAdapter(logger, {
        'interaction_id': 'no_id',
        'pipeline_name': 'no_pipeline'
    })
    
    return logger

def setup_app_logging(config: dict = None, default_level=logging.INFO) -> None:
    """
    Set up logging for the entire application.

    This function configures the root logger with handlers for console
    and file output. Log levels and file paths can be specified via
    the optional config dictionary.

    Args:
        config (dict, optional): A dictionary containing logging configurations.
                                Expected keys:
                                - 'level': String representation of log level (e.g., "DEBUG", "INFO").
                                - 'file_path': Path to the log file (e.g., "ai_assistant.log").
                                - 'max_bytes': Max size of the log file before rotation.
                                - 'backup_count': Number of backup log files to keep.
                                - 'format': Custom log format string.
                                - 'date_format': Custom log date format string.
        default_level (int, optional): The default logging level if not specified
                                     in the config. Defaults to logging.INFO.
    """
    if config is None:
        config = {}

    # Determine log level
    log_level_str = config.get('level', str(logging.getLevelName(default_level))).upper()
    numeric_log_level = getattr(logging, log_level_str, default_level)
    if not isinstance(numeric_log_level, int):
        print(f"Warning: Invalid log level string '{log_level_str}'. Using default level {logging.getLevelName(default_level)}.", file=sys.stderr)
        numeric_log_level = default_level

    # Get log format and date format
    log_format = config.get('format', DEFAULT_LOG_FORMAT)
    log_date_format = config.get('date_format', DEFAULT_LOG_DATE_FORMAT)
    
    # Create structured formatter
    formatter = StructuredLogFormatter(log_format, datefmt=log_date_format)

    # Get root logger
    # It's generally recommended to configure the root logger or specific application loggers.
    # Configuring the root logger makes it easy for all modules using logging.getLogger(__name__)
    # to inherit this configuration.
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_log_level)

    # Remove any existing handlers
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

    # Console Handler (StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File Handler (RotatingFileHandler)
    log_file_path = config.get('file_path', 'ai_assistant.log')
    if log_file_path:
        try:
            max_bytes = int(config.get('max_bytes', 5*1024*1024))  # 5 MB
            backup_count = int(config.get('backup_count', 3))       # Keep 3 backup files

            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path, 
                maxBytes=max_bytes, 
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            print(f"Logging to file: {log_file_path} with level {log_level_str}", file=sys.stdout)
        except Exception as e:
            print(f"Error setting up file logging to {log_file_path}: {e}. File logging will be disabled.", file=sys.stderr)
    else:
        print("File logging is disabled as no 'file_path' was provided in logging config.", file=sys.stdout)
    
    # Initial log message to confirm setup
    # Using a temporary logger here to avoid issues if getLogger(__name__) is used before basicConfig is called on root
    # However, since we are configuring the root logger directly, this should be fine.
    initial_logger = get_logger("LoggingConfig")
    initial_logger.info("Application logging setup complete. Level: %s", log_level_str)

