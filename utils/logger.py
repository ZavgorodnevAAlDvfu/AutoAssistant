# logger_config.py
import logging
import os
from dotenv import load_dotenv
import colorlog

load_dotenv()

def setup_logger(name):
    """Настраивает и возвращает логгер с цветным выводом в консоль и записью в файл."""
    log_level = os.environ.get("LOGGING_LEVEL", "INFO").upper()
    log_file = os.environ.get("LOG_FILE", "app.log")

    log_colors = {
        'DEBUG': 'white',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors=log_colors
    ))

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
