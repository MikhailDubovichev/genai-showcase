"""
Core metrics and monitoring decorators for the AI Assistant.

This module defines Prometheus metrics and decorators for tracking:
- Request latency and counts
- Error rates
- Pipeline processing time
- Tool execution time
- External API latency (LLM and Integrator)
"""

import time
import functools
import logging
from typing import Optional, Callable, Union, Dict
from prometheus_client import Counter, Histogram

# Configure logger
logger = logging.getLogger(__name__)

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]  # Define buckets in seconds
)

# Error metrics
ERROR_COUNT = Counter(
    'error_total',
    'Total number of errors',
    ['type', 'location']  # type: e.g., 'http', 'pipeline', 'tool'; location: specific component
)

# Pipeline metrics
PIPELINE_PROCESSING_TIME = Histogram(
    'pipeline_processing_duration_seconds',
    'Time spent processing in pipeline',
    ['pipeline_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
)

# Tool execution metrics
TOOL_EXECUTION_TIME = Histogram(
    'tool_execution_duration_seconds',
    'Time spent executing tools',
    ['tool_name'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, float("inf")]
)

# External API metrics
LLM_REQUEST_TIME = Histogram(
    'llm_request_duration_seconds',
    'Time spent waiting for LLM API',
    ['model'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
)

INTEGRATOR_REQUEST_TIME = Histogram(
    'integrator_request_duration_seconds',
    'Time spent waiting for Smart Home Integrator API',
    ['endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, float("inf")]
)

def track_latency(metric: Histogram, labels: Optional[Callable] = None) -> Callable:
    """
    A decorator factory that tracks the execution time of a function using a Prometheus Histogram.
    
    Args:
        metric (Histogram): The Prometheus Histogram to record the timing in
        labels (Callable, optional): Function that returns metric labels dictionary
        
    Returns:
        Callable: The decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if labels and args:
                    # For instance methods, first arg is 'self'
                    label_dict = labels(args[0])
                    metric.labels(**label_dict).observe(duration)
                else:
                    metric.observe(duration)
                
                # Log timing information at debug level
                func_name = func.__name__
                logger.debug(
                    f"Function {func_name} execution time: {duration:.2f} seconds",
                    extra={'duration': duration, 'function': func_name}
                )
        return wrapper
    return decorator

def track_errors(error_type: str, location: str) -> Callable:
    """
    A decorator factory that tracks errors occurring in a function.
    
    Args:
        error_type (str): Type of error (e.g., 'http', 'pipeline', 'tool')
        location (str): Where the error occurred
        
    Returns:
        Callable: The decorated function
        
    Example:
        @track_errors('pipeline', 'device_control')
        def process_query(self, message: str):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Increment error counter
                ERROR_COUNT.labels(
                    type=error_type,
                    location=location
                ).inc()
                
                # Log error with context
                logger.error(
                    f"Error in {location} ({error_type}): {str(e)}",
                    extra={
                        'error_type': error_type,
                        'location': location,
                        'error': str(e)
                    },
                    exc_info=True
                )
                raise  # Re-raise the exception after tracking
        return wrapper
    return decorator 