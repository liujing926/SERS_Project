#!/usr/bin/env python
# coding: utf-8

"""Logging utility module"""
import logging
import os
from datetime import datetime


def setup_logger(log_dir='./experiments/logs'):
    """Set up the logger."""
    os.makedirs(log_dir, exist_ok=True)

    log_filename = datetime.now().strftime('%Y%m%d_%H%M%S.log')
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),  # Write to file
            logging.StreamHandler()         # Output to console
        ]
    )
    return logging.getLogger(__name__)