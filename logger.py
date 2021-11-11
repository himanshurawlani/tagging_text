"""
Module to create custom logger for tagging text pipeline
"""

import os
import sys
import logging


def create_logger(module_name, log_dir):
    """Creates python logger

    Arguments:
        module_name {str}: Unique name of the module
        log_dir {str}: Path of directory to log

    Returns:
        logger: logger object
    """

    # create logger with 'module_name'
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    os.makedirs(f"{log_dir}", exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "app.log"))
    file_handler.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    verbose_formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d - %(levelname)s - %(process)d:%(thread)d - %(filename)s - %(funcName)s:%(lineno)d] %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
    )
    file_handler.setFormatter(verbose_formatter)
    console_handler.setFormatter(verbose_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
