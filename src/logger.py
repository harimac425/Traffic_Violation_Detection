import logging
import os
from datetime import datetime
from pathlib import Path

# Log directory
LOG_DIR = Path("logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Main log file name with timestamp for each session
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = LOG_DIR / f"app_{timestamp}.log"

# Unified logger configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def get_logger(name):
    """Returns a logger instance with the given name."""
    return logging.getLogger(name)

# Initial entry
logger = get_logger("App")
logger.info("=" * 60)
logger.info("  TRAFFIC VIOLATION DETECTION SYSTEM - LOG SESSION STARTED")
logger.info("=" * 60)
logger.info(f"Log file: {log_file}")
