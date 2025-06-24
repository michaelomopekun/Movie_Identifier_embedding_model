import logging
import sys
from pathlib import Path

log_dir = Path(__file__).resolve().parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

log_file = log_dir / "app.log"

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# File handler with utf-8 encoding
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Console handler with utf-8 encoding
console_handler = logging.StreamHandler(sys.stdout)
try:
    console_handler.stream.reconfigure(encoding="utf-8")
except AttributeError:
    pass
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Main logger
logger = logging.getLogger("embedding_service")
logger.setLevel(logging.INFO)

# Avoid duplicate handlers during reload
if not logger.hasHandlers():
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)