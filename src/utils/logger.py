"""Structured logging with sensitive data redaction."""

import json
import logging
from datetime import datetime
from typing import Any, Dict

SENSITIVE_KEYS = {"password", "token", "api_key", "secret", "auth", "key"}


class SecureJSONFormatter(logging.Formatter):
    """JSON formatter that redacts sensitive information."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON with sensitive data redaction.

        Args:
            record: LogRecord to format

        Returns:
            JSON string representation of log record
        """
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add module, function, line number
        if record.module:
            log_data["module"] = record.module
        if record.funcName:
            log_data["function"] = record.funcName
        if record.lineno:
            log_data["line"] = record.lineno

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields with redaction
        extra = getattr(record, "extra", {})
        if extra:
            redacted_extra = self._redact_sensitive(extra)
            log_data.update(redacted_extra)

        return json.dumps(log_data)

    def _redact_sensitive(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redact sensitive keys from log data.

        Args:
            data: Dictionary with potentially sensitive data

        Returns:
            Dictionary with sensitive values redacted
        """
        redacted = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in SENSITIVE_KEYS):
                redacted[key] = "***REDACTED***"
            elif isinstance(value, dict):
                redacted[key] = self._redact_sensitive(value)
            else:
                redacted[key] = value
        return redacted


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with secure JSON formatting.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Add console handler with JSON formatter
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(SecureJSONFormatter())

    logger.addHandler(handler)
    logger.propagate = False

    return logger
