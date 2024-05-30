import logging
from time import strftime
from pathlib import Path


def log(name: str = 'reezodata'):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    logger = logging.getLogger(name)
    Path('logs').mkdir(parents=True, exist_ok=True)
    log_filename = 'logs/log_' + strftime('%Y-%m-%d_%H-%M-%S') + '.txt'
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
