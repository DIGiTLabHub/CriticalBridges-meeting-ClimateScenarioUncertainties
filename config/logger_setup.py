"""
Logging configuration for project.
"""
from pathlib import Path
from datetime import datetime
import logging

def setup_logging(log_level=logging.INFO, log_to_file=True):
    """
    Configure logging for entire project.

    Parameters
    ----------
    log_level : int, optional
        Logging level (default: logging.INFO)
    log_to_file : bool, optional
        Whether to log to file (default: True)

    Returns
    -------
    None
        Sets up logging configuration
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    handlers = [logging.StreamHandler()]

    if log_to_file:
        from .paths import PATHS
        log_dir = PATHS['output_data'] / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )


def get_logger(name: str):
    """
    Get a logger with specified name.

    Parameters
    ----------
    name : str
        Logger name (typically __name__)

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    return logging.getLogger(name)
