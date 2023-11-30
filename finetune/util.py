
import logging

def create_file_logger(file_name: str = 'log.txt', log_format: str = '%(message)s', log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    handler = logging.FileHandler(file_name)
    handler.setLevel(log_level)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(log_level)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger