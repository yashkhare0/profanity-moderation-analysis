import logging
import sys

def setup_logging():
    """
    Set up logging configuration.
    """
    logger = logging.getLogger('CurseWordsModeration')
    logger.setLevel(logging.DEBUG)
    
    # Console Handler with UTF-8 encoding
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    console_handler.setStream(sys.stdout)
    console_handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
    
    # File Handler with UTF-8 encoding
    file_handler = logging.FileHandler('project.log', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    
    # Add Handlers to Logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger